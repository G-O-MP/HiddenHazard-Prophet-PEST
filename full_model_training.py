# -*- coding: utf-8 -*-
"""
完整训练脚本 - 整合原始模型 + 所有改进方法

原始模型：FewShotCamouflageSegWithClass
改进方法：
1. 数据增强 (MixUp, CutMix, 几何变换)
2. 注意力机制 (CBAM, SE, ECA)
3. 多尺度融合 (ASPP, FPN)
4. 高级损失函数 (Focal Loss, Tversky Loss, Dice Loss)
5. 深度监督
6. 测试时增强 (TTA)
"""
import os
import sys
import json
import time
import random
import warnings
import math
from datetime import datetime
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms

RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiment_results", "real_training")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        w = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * w


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k_size = int(abs((math.log2(channels) + b) / gamma))
        k_size = k_size if k_size % 2 else k_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        ca = self.channel_att(x).view(x.size(0), -1, 1, 1)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        return x * sa


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.project = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU())
    
    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=True)
        return self.project(torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1))


class ResNetEncoder(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        if backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feat_dims = {'c2': 256, 'c3': 512, 'c4': 1024, 'c5': 2048}
        else:
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            self.feat_dims = {'c2': 64, 'c3': 128, 'c4': 256, 'c5': 512}
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.cbam2 = CBAM(self.feat_dims['c2'])
        self.cbam3 = CBAM(self.feat_dims['c3'])
        self.cbam4 = CBAM(self.feat_dims['c4'])
        self.cbam5 = CBAM(self.feat_dims['c5'])
        
        self.se4 = SEBlock(self.feat_dims['c4'])
        self.eca5 = ECABlock(self.feat_dims['c5'])
    
    def forward(self, x):
        feats = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        feats['c2'] = self.cbam2(x)
        
        x = self.layer2(x)
        feats['c3'] = self.cbam3(x)
        
        x = self.layer3(x)
        x = self.se4(x)
        feats['c4'] = self.cbam4(x)
        
        x = self.layer4(x)
        feats['c5'] = self.eca5(x)
        
        return feats


class MaskedAvgPool(nn.Module):
    def forward(self, feat, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = F.interpolate(mask.float(), size=feat.shape[-2:], mode='nearest')
        masked = feat * mask
        return masked.sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-6)


class ProtoMatcher(nn.Module):
    def forward(self, feat, proto):
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1)
        
        if proto.dim() == 4:
            proto = proto.view(proto.size(0), C, -1).mean(dim=-1)
        
        proto_expanded = proto.view(1, C, 1).expand(B, -1, -1)
        
        sim = F.cosine_similarity(feat_flat, proto_expanded, dim=1)
        return sim.view(B, H, W).unsqueeze(1)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, q, k, v):
        B, N_q, C = q.shape
        _, N_kv, _ = k.shape
        
        q = self.q_proj(q).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).contiguous().view(B, N_q, C)
        return self.proj(x)


class ProtoFormerBridge(nn.Module):
    def __init__(self, dim, num_heads=8, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.cross_attn = CrossAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, q_c4, s_c4, s_mask):
        B, C, H, W = q_c4.shape
        
        q_tokens = q_c4.flatten(2).transpose(1, 2)
        
        if s_mask.dim() == 3:
            s_mask = s_mask.unsqueeze(1)
        s_mask = F.interpolate(s_mask.float(), size=s_c4.shape[-2:], mode='nearest')
        
        fg_proto = (s_c4 * s_mask).sum(dim=(2, 3)) / (s_mask.sum(dim=(2, 3)) + 1e-6)
        bg_proto = (s_c4 * (1 - s_mask)).sum(dim=(2, 3)) / ((1 - s_mask).sum(dim=(2, 3)) + 1e-6)
        
        fg_proto = fg_proto.mean(dim=0, keepdim=True)
        bg_proto = bg_proto.mean(dim=0, keepdim=True)
        
        proto_tokens = torch.stack([fg_proto, bg_proto], dim=1).expand(B, -1, -1)
        
        enhanced = self.cross_attn(q_tokens, proto_tokens, proto_tokens)
        enhanced = self.norm1(q_tokens + self.res_scale * enhanced)
        enhanced = self.norm2(enhanced + self.res_scale * self.mlp(enhanced))
        
        enhanced = enhanced.transpose(1, 2).view(B, C, H, W)
        return q_c4 + self.res_scale * (enhanced - q_c4)


class C2FDecoder(nn.Module):
    def __init__(self, ch_c2=256, ch_c3=512, ch_c4=1024, ch_c5=2048, mid=256):
        super().__init__()
        
        self.aspp = ASPP(ch_c5, mid)
        
        self.up5 = nn.Sequential(nn.Conv2d(mid, mid, 3, 1, 1), nn.BatchNorm2d(mid), nn.ReLU())
        self.up4 = nn.Sequential(nn.Conv2d(ch_c4 + mid, mid, 3, 1, 1), nn.BatchNorm2d(mid), nn.ReLU())
        self.up3 = nn.Sequential(nn.Conv2d(ch_c3 + mid, mid, 3, 1, 1), nn.BatchNorm2d(mid), nn.ReLU())
        self.up2 = nn.Sequential(nn.Conv2d(ch_c2 + mid, mid, 3, 1, 1), nn.BatchNorm2d(mid), nn.ReLU())
        
        self.final = nn.Conv2d(mid, 1, 1)
        self.aux = nn.Conv2d(mid, 1, 1)
        self.guide_conv = nn.Conv2d(3, mid, 1)
    
    def forward(self, feats, guides):
        g = self.guide_conv(guides['g5'])
        
        d = self.aspp(feats['c5'])
        d = self.up5(d)
        d = F.interpolate(d, size=feats['c4'].shape[-2:], mode='bilinear', align_corners=True)
        
        aux4 = self.aux(d)
        
        d = self.up4(torch.cat([d, feats['c4']], dim=1))
        d = F.interpolate(d, size=feats['c3'].shape[-2:], mode='bilinear', align_corners=True)
        
        aux3 = self.aux(d)
        
        d = self.up3(torch.cat([d, feats['c3']], dim=1))
        d = F.interpolate(d, size=feats['c2'].shape[-2:], mode='bilinear', align_corners=True)
        
        d = self.up2(torch.cat([d, feats['c2']], dim=1))
        
        return {'log2': self.final(d), 'log3': aux3, 'log4': aux4}


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.fc(x)


class BoundingBoxHead(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4)
        )
    
    def forward(self, mask_prob):
        x = self.conv(mask_prob).flatten(1)
        return torch.sigmoid(self.fc(x))


class FewShotCamouflageSegModel(nn.Module):
    def __init__(self, num_classes=102, backbone='resnet50', pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        self.encoder = ResNetEncoder(backbone=backbone, pretrained=pretrained)
        feat_dims = self.encoder.feat_dims
        
        self.pool = MaskedAvgPool()
        self.match = ProtoMatcher()
        self.cross_attn = ProtoFormerBridge(dim=feat_dims['c4'], num_heads=8)
        
        self.decoder = C2FDecoder(
            ch_c2=feat_dims['c2'],
            ch_c3=feat_dims['c3'],
            ch_c4=feat_dims['c4'],
            ch_c5=feat_dims['c5']
        )
        
        self.class_head = ClassificationHead(feat_dims['c4'], num_classes)
        self.bbox_head = BoundingBoxHead()
    
    def build_protos(self, support_feats, support_mask):
        protos = {}
        M = support_mask.float()
        
        for lvl in ["c2", "c3", "c4", "c5"]:
            Fs = support_feats[lvl]
            p_fg = self.pool(Fs, M).mean(dim=0, keepdim=True)
            p_bg = self.pool(Fs, 1.0 - M).mean(dim=0, keepdim=True)
            protos[lvl] = {"p_fg": p_fg, "p_bg": p_bg}
        
        return protos
    
    def build_guides(self, query_feats, protos):
        guides = {}
        for lvl in ["c2", "c3", "c4", "c5"]:
            Fq = query_feats[lvl]
            sim_fg = self.match(Fq, protos[lvl]["p_fg"])
            sim_bg = self.match(Fq, protos[lvl]["p_bg"])
            guides[f'g{lvl[1]}'] = torch.cat([sim_fg - sim_bg, sim_fg, (sim_fg + sim_bg) / 2], dim=1)
        
        return guides
    
    def forward(self, support_imgs, support_masks, query_img):
        if support_masks.dim() == 3:
            support_masks = support_masks.unsqueeze(1)
        
        s_feats = self.encoder(support_imgs)
        q_feats = self.encoder(query_img)
        
        q_feats['c4'] = self.cross_attn(q_feats['c4'], s_feats['c4'], support_masks)
        
        protos = self.build_protos(s_feats, support_masks)
        guides = self.build_guides(q_feats, protos)
        
        logits = self.decoder(q_feats, guides)
        
        seg_logits = F.interpolate(logits["log2"], size=query_img.shape[-2:], mode="bilinear", align_corners=False)
        seg_prob = torch.sigmoid(seg_logits)
        
        class_logits = self.class_head(q_feats['c4'])
        bbox = self.bbox_head(seg_prob)
        
        return {
            'seg_logits': seg_logits,
            'seg_prob': seg_prob,
            'class_logits': class_logits,
            'bbox': bbox,
            'logits': logits
        }


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        return 1 - (tp + 1e-6) / (tp + self.alpha * fp + self.beta * fn + 1e-6)


class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum()
        return 1 - (2 * inter + 1e-6) / (pred.sum() + target.sum() + 1e-6)


class CombinedSegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        return 0.4 * self.focal(pred, target) + 0.3 * self.tversky(pred, target) + 0.3 * self.dice(pred, target)


class DeepSupervisionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = {'log2': 1.0, 'log3': 0.5, 'log4': 0.25}
        self.criterion = CombinedSegLoss()
    
    def forward(self, logits, target):
        total_loss = 0
        for lvl, weight in self.weights.items():
            if lvl in logits:
                logit = logits[lvl]
                target_resized = F.interpolate(target.unsqueeze(1), size=logit.shape[-2:], mode='nearest').squeeze(1)
                total_loss += weight * self.criterion(logit.squeeze(1), target_resized)
        return total_loss


class MixUp:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
    
    def __call__(self, images, targets):
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(images.size(0))
        return lam * images + (1 - lam) * images[idx], lam * targets + (1 - lam) * targets[idx]


class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, images, targets):
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(images.size(0))
        
        H, W = images.size(2), images.size(3)
        cut_w, cut_h = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))
        cx, cy = np.random.randint(W), np.random.randint(H)
        
        bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
        bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
        
        images[:, :, bbx1:bbx2, bby1:bby2] = images[idx, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (W * H)
        
        return images, lam * targets + (1 - lam) * targets[idx]


class AdvancedAugmentation:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, gt=None):
        if random.random() < self.p:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if gt is not None:
                    gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                if gt is not None:
                    gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
            
            if random.random() < 0.3:
                angle = random.uniform(-15, 15)
                img = img.rotate(angle)
                if gt is not None:
                    gt = gt.rotate(angle)
            
            if random.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            
            if random.random() < 0.3:
                factor = random.uniform(0.8, 1.2)
                img = Image.eval(img, lambda x: min(255, max(0, int(x * factor))))
        
        return img, gt


class COD10KDataset(Dataset):
    def __init__(self, root_dir, img_size=256, max_samples=None, augment=False):
        self.img_size = img_size
        self.augment = augment
        self.aug = AdvancedAugmentation(p=0.5)
        
        self.img_dir = os.path.join(root_dir, 'Test', 'Image')
        self.gt_dir = os.path.join(root_dir, 'Test', 'GT_Object')
        
        self.samples = []
        for fname in os.listdir(self.img_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.img_dir, fname)
                gt_name = fname.replace('.jpg', '.png').replace('.jpeg', '.png')
                gt_path = os.path.join(self.gt_dir, gt_name)
                
                if os.path.exists(gt_path):
                    self.samples.append({'img_path': img_path, 'gt_path': gt_path, 'name': fname})
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img_path']).convert("RGB")
        gt = Image.open(sample['gt_path']).convert("L")
        
        if self.augment:
            img, gt = self.aug(img, gt)
        
        img = img.resize((self.img_size, self.img_size))
        gt = gt.resize((self.img_size, self.img_size))
        
        img = self.normalize(self.to_tensor(img))
        gt = torch.from_numpy((np.array(gt) > 127).astype(np.float32))
        
        return {'image': img, 'gt': gt, 'name': sample['name']}


class IP02Dataset(Dataset):
    def __init__(self, root_dir, img_size=224, max_samples=None, augment=False):
        self.img_size = img_size
        self.augment = augment
        
        det_dir = os.path.join(root_dir, "Detection-20260318T092509Z-1-001", "Detection", "VOC2007")
        self.img_dir = os.path.join(det_dir, "JPEGImages", "JPEGImages")
        
        self.samples = []
        for fname in os.listdir(self.img_dir):
            if fname.lower().endswith('.jpg'):
                img_id = fname.replace('.jpg', '')
                if img_id.startswith('IP'):
                    try:
                        class_id = int(img_id[2:5]) - 1
                        if 0 <= class_id < 102:
                            self.samples.append({'img_path': os.path.join(self.img_dir, fname), 'class_id': class_id})
                    except:
                        pass
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        self.num_classes = max(s['class_id'] for s in self.samples) + 1 if self.samples else 102
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if self.augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img_path']).convert("RGB").resize((self.img_size, self.img_size))
        
        if self.augment:
            img = self.aug_transform(img)
        
        return {'image': self.normalize(self.to_tensor(img)), 'class_id': sample['class_id']}


def compute_seg_metrics(pred, gt):
    pred, gt = pred.flatten(), gt.flatten()
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {'mIoU': float(iou), 'Dice': float(dice), 'Precision': float(precision), 'Recall': float(recall), 'F1': float(f1)}


def compute_cls_metrics(preds, labels, num_classes):
    preds, labels = np.array(preds), np.array(labels)
    acc = np.mean(preds == labels)
    
    per_class_f1 = []
    for c in range(min(num_classes, max(labels) + 1)):
        mask = labels == c
        if mask.sum() > 0:
            prec = np.mean(preds[mask] == c) if (preds == c).sum() > 0 else 0
            rec = np.mean(preds[mask] == c)
            f1 = 2 * prec * rec / (prec + rec + 1e-6) if (prec + rec) > 0 else 0
            per_class_f1.append(f1)
    
    return {'Accuracy': float(acc), 'MacroF1': float(np.mean(per_class_f1) if per_class_f1 else 0)}


def tta_predict(model, support_imgs, support_masks, query_imgs, scales=[0.9, 1.0, 1.1]):
    model.eval()
    B, H, W = query_imgs.size(0), query_imgs.size(2), query_imgs.size(3)
    preds = []
    
    def make_divisible(h, w, divisor=32):
        new_h = (h // divisor) * divisor
        new_w = (w // divisor) * divisor
        return max(new_h, divisor), max(new_w, divisor)
    
    with torch.no_grad():
        for scale in scales:
            new_h, new_w = make_divisible(int(H * scale), int(W * scale))
            
            scaled_q = F.interpolate(query_imgs, (new_h, new_w), mode='bilinear', align_corners=True)
            scaled_s = F.interpolate(support_imgs, (new_h, new_w), mode='bilinear', align_corners=True)
            if support_masks.dim() == 3:
                scaled_m = F.interpolate(support_masks.unsqueeze(1), (new_h, new_w), mode='nearest').squeeze(1)
            else:
                scaled_m = F.interpolate(support_masks, (new_h, new_w), mode='nearest')
            
            pred = model(scaled_s, scaled_m, scaled_q)['seg_prob']
            pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)
            preds.append(pred)
            
            pred_f = model(scaled_s, scaled_m, torch.flip(scaled_q, [3]))['seg_prob']
            pred_f = torch.flip(pred_f, [3])
            pred_f = F.interpolate(pred_f, (H, W), mode='bilinear', align_corners=True)
            preds.append(pred_f)
    
    return torch.stack(preds).mean(dim=0)


def train_seg_model(train_loader, val_loader, epochs=30, lr=1e-4):
    print(f"\n  初始化小样本伪装目标分割模型...")
    print(f"    - ResNet50骨干网络 (ImageNet预训练)")
    print(f"    - CBAM + SE + ECA 三重注意力")
    print(f"    - ASPP多尺度融合")
    print(f"    - 原型匹配 + Cross-Attention")
    print(f"    - FPN解码器 + 深度监督")
    print(f"    - 分类头 + 边界框头")
    
    model = FewShotCamouflageSegModel(num_classes=102, backbone='resnet50', pretrained=True).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    seg_criterion = DeepSupervisionLoss()
    
    mixup = MixUp(alpha=0.4)
    cutmix = CutMix(alpha=1.0)
    
    best_miou = 0
    best_metrics = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            images = batch['image'].to(DEVICE)
            gts = batch['gt'].to(DEVICE)
            
            if random.random() < 0.25:
                images, gts = mixup(images, gts)
            elif random.random() < 0.25:
                images, gts = cutmix(images, gts)
            
            K = min(5, images.size(0) // 2)
            if K < 1 or images.size(0) < 2:
                continue
            
            support_imgs = images[:K]
            support_masks = gts[:K]
            query_imgs = images[K:]
            query_gts = gts[K:]
            
            if query_imgs.size(0) == 0:
                continue
            
            optimizer.zero_grad()
            output = model(support_imgs, support_masks, query_imgs)
            loss = seg_criterion(output['logits'], query_gts)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        
        model.eval()
        all_metrics = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                gts = batch['gt'].numpy()
                
                K = min(3, images.size(0) // 2)
                if K < 1 or images.size(0) < 2:
                    continue
                
                support_imgs = images[:K]
                support_masks = torch.ones(K, images.size(2), images.size(3), device=DEVICE)
                query_imgs = images[K:]
                query_gts = gts[K:]
                
                if query_imgs.size(0) == 0:
                    continue
                
                preds = tta_predict(model, support_imgs, support_masks, query_imgs).cpu().numpy()
                preds = (preds > 0.5).astype(np.float32)
                
                for i in range(query_imgs.size(0)):
                    all_metrics.append(compute_seg_metrics(preds[i, 0], query_gts[i]))
        
        if all_metrics:
            avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
            
            if avg_metrics['mIoU'] > best_miou:
                best_miou = avg_metrics['mIoU']
                best_metrics = avg_metrics.copy()
            
            print(f"    Epoch {epoch+1}/{epochs}: Loss={train_loss/max(1,num_batches):.4f}, mIoU={avg_metrics['mIoU']:.4f}, Dice={avg_metrics['Dice']:.4f}, F1={avg_metrics['F1']:.4f}")
        else:
            print(f"    Epoch {epoch+1}/{epochs}: Loss={train_loss/max(1,num_batches):.4f}")
    
    return best_metrics or {'mIoU': 0, 'Dice': 0, 'Precision': 0, 'Recall': 0, 'F1': 0}


def train_cls_model(train_loader, val_loader, num_classes, epochs=30, lr=1e-4):
    print(f"\n  初始化改进分类模型...")
    print(f"    - ResNet50骨干网络 (ImageNet预训练)")
    print(f"    - CBAM + SE + ECA 三重注意力")
    
    class ClsModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
            self.cbam = CBAM(2048)
            self.se = SEBlock(2048)
            self.eca = ECABlock(2048)
            self.fc = nn.Linear(2048, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = self.cbam(x)
            x = self.se(x)
            x = self.eca(x)
            return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
    
    model = ClsModel(num_classes).to(DEVICE)
    
    for name, param in model.named_parameters():
        param.requires_grad = 'layer4' in name or 'fc' in name or 'cbam' in name or 'se' in name or 'eca' in name
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0
    best_metrics = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            images = batch['image'].to(DEVICE)
            labels = batch['class_id'].to(DEVICE)
            
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                labels = batch['class_id'].numpy()
                preds = model(images).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        metrics = compute_cls_metrics(all_preds, all_labels, num_classes)
        
        if metrics['Accuracy'] > best_acc:
            best_acc = metrics['Accuracy']
            best_metrics = metrics.copy()
        
        print(f"    Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.4f}, Acc={metrics['Accuracy']:.4f}, MacroF1={metrics['MacroF1']:.4f}")
    
    return best_metrics


def main():
    print("="*80)
    print("完整训练实验 - 原始模型 + 所有改进方法")
    print(f"设备: {DEVICE}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\n原始模型架构:")
    print("  - FewShotCamouflageSeg (原型匹配 + Cross-Attention)")
    print("  - ResNet50骨干网络 (ImageNet预训练)")
    print("  - 分类头 + 边界框头")
    
    print("\n改进方法:")
    print("  1. 注意力机制: CBAM + SE + ECA 三重注意力")
    print("  2. 多尺度融合: ASPP + FPN")
    print("  3. 数据增强: MixUp, CutMix, 几何变换, 颜色抖动")
    print("  4. 损失函数: Focal Loss + Tversky Loss + Dice Loss")
    print("  5. 深度监督")
    print("  6. 测试时增强 (TTA)")
    print("  7. 标签平滑")
    print("  8. 学习率调度: CosineAnnealingWarmRestarts")
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': DEVICE,
        'model': 'FewShotCamouflageSeg + All Improvements',
        'experiments': {}
    }
    
    cod10k_root = r"d:\University\竞赛\4C\智慧农林\模型代码\COD10K-v3\COD10K-v3"
    ip02_root = r"d:\University\竞赛\4C\智慧农林\模型代码\数据集\数据集\IP02"
    
    print("\n" + "="*80)
    print("实验1: COD10K 伪装目标分割")
    print("="*80)
    
    print(f"\n[1] 加载数据集...")
    # img_size=256 is already a multiple of 32 (256=32*8)
    full_dataset = COD10KDataset(cod10k_root, img_size=256, augment=True)
    print(f"    总样本数: {len(full_dataset)}")
    
    train_size = int(0.8 * len(full_dataset))
    train_dataset, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"    训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    print(f"\n[2] 开始训练 (30 epochs)...")
    seg_metrics = train_seg_model(train_loader, val_loader, epochs=30, lr=1e-4)
    results['experiments']['COD10K_Segmentation'] = seg_metrics
    
    print("\n" + "="*80)
    print("实验2: IP02 害虫分类")
    print("="*80)
    
    print(f"\n[1] 加载数据集...")
    full_dataset = IP02Dataset(ip02_root, img_size=224, augment=True)
    print(f"    总样本数: {len(full_dataset)}")
    print(f"    类别数: {full_dataset.num_classes}")
    
    train_size = int(0.8 * len(full_dataset))
    train_dataset, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"    训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    print(f"\n[2] 开始训练 (30 epochs)...")
    cls_metrics = train_cls_model(train_loader, val_loader, full_dataset.num_classes, epochs=30, lr=1e-4)
    results['experiments']['IP02_Classification'] = cls_metrics
    
    results_file = os.path.join(RESULTS_DIR, 'full_model_training_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("训练完成!")
    print("="*80)
    print(f"\nCOD10K分割结果:")
    for k, v in seg_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nIP02分类结果:")
    for k, v in cls_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\n结果已保存到: {results_file}")
    
    return results


if __name__ == "__main__":
    results = main()
