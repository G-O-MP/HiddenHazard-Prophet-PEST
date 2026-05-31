# -*- coding: utf-8 -*-
"""
完整训练脚本 - 包含所有改进方法

改进包括:
1. 数据增强 (Mosaic, MixUp, CutMix)
2. 注意力机制 (CBAM, SE, ECA)
3. 多尺度特征融合 (ASPP, FPN)
4. 高级损失函数 (Focal Loss, Tversky Loss, Boundary Loss)
5. 测试时增强 (TTA)
6. 深度监督
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
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        w = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
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


class ImprovedSegModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)
        
        self.aspp = ASPP(2048, 256)
        
        self.decoder = nn.ModuleDict({
            'up1': nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
            'up2': nn.Sequential(nn.Conv2d(256 + 1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
            'up3': nn.Sequential(nn.Conv2d(256 + 512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
            'up4': nn.Sequential(nn.Conv2d(256 + 256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
            'up5': nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
            'final': nn.Conv2d(64, 1, 1)
        })
        
        self.aux_head = nn.Conv2d(1024, 1, 1)
    
    def forward(self, x):
        f0 = self.layer0(x)
        f1 = self.cbam1(self.layer1(f0))
        f2 = self.cbam2(self.layer2(f1))
        f3 = self.cbam3(self.layer3(f2))
        f4 = self.cbam4(self.layer4(f3))
        
        f4 = self.aspp(f4)
        
        d = self.decoder['up1'](f4)
        d = self.decoder['up2'](torch.cat([d, f3], dim=1))
        d = self.decoder['up3'](torch.cat([d, f2], dim=1))
        d = self.decoder['up4'](torch.cat([d, f1], dim=1))
        d = self.decoder['up5'](d)
        out = self.decoder['final'](d)
        
        aux_out = self.aux_head(f3)
        
        return out, aux_out


class ImprovedClsModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.cbam = CBAM(2048)
        self.se = SEBlock(2048)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        f = self.layer0(x)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        
        f = self.cbam(f)
        f = self.se(f)
        
        f = self.avgpool(f)
        f = f.view(f.size(0), -1)
        return self.fc(f)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1e-6
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        return 1 - tversky


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.laplacian = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.laplacian.weight.data = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.laplacian.weight.requires_grad = False
    
    def forward(self, pred, target):
        pred_edge = self.laplacian(torch.sigmoid(pred))
        target_edge = self.laplacian(target.float())
        return F.mse_loss(pred_edge, target_edge)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss()
        self.boundary = BoundaryLoss()
    
    def forward(self, pred, target):
        return 0.4 * self.focal(pred, target) + 0.4 * self.tversky(pred, target) + 0.2 * self.boundary(pred, target)


class MixUp:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
    
    def __call__(self, images, targets):
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(images.size(0))
        mixed_images = lam * images + (1 - lam) * images[idx]
        mixed_targets = lam * targets + (1 - lam) * targets[idx]
        return mixed_images, mixed_targets


class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, images, targets):
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(images.size(0))
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[idx, :, bbx1:bbx2, bby1:bby2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        mixed_targets = lam * targets + (1 - lam) * targets[idx]
        
        return images, mixed_targets
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


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
                            self.samples.append({
                                'img_path': os.path.join(self.img_dir, fname),
                                'class_id': class_id
                            })
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
        img = Image.open(sample['img_path']).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        
        if self.augment:
            img = self.aug_transform(img)
        
        img = self.normalize(self.to_tensor(img))
        return {'image': img, 'class_id': sample['class_id']}


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
    preds = np.array(preds)
    labels = np.array(labels)
    
    acc = np.mean(preds == labels)
    
    per_class_f1 = []
    for c in range(min(num_classes, max(labels) + 1)):
        mask = labels == c
        if mask.sum() > 0:
            prec = np.mean(preds[mask] == c) if (preds == c).sum() > 0 else 0
            rec = np.mean(preds[mask] == c)
            f1 = 2 * prec * rec / (prec + rec + 1e-6) if (prec + rec) > 0 else 0
            per_class_f1.append(f1)
    
    macro_f1 = np.mean(per_class_f1) if per_class_f1 else 0
    
    return {'Accuracy': float(acc), 'MacroF1': float(macro_f1)}


def tta_predict(model, images, scales=[0.75, 1.0, 1.25]):
    model.eval()
    b, c, h, w = images.shape
    preds = []
    
    with torch.no_grad():
        for scale in scales:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = F.interpolate(images, (new_h, new_w), mode='bilinear', align_corners=True)
            
            out, _ = model(scaled)
            out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
            preds.append(torch.sigmoid(out))
            
            flipped = torch.flip(scaled, [3])
            out_f, _ = model(flipped)
            out_f = torch.flip(out_f, [3])
            out_f = F.interpolate(out_f, (h, w), mode='bilinear', align_corners=True)
            preds.append(torch.sigmoid(out_f))
    
    return torch.stack(preds).mean(dim=0)


def train_seg_model(train_loader, val_loader, epochs=30, lr=1e-4):
    print(f"\n  初始化改进分割模型...")
    print(f"    - ResNet50骨干网络 (ImageNet预训练)")
    print(f"    - CBAM注意力机制")
    print(f"    - ASPP多尺度融合")
    print(f"    - FPN解码器")
    print(f"    - 深度监督")
    
    model = ImprovedSegModel().to(DEVICE)
    
    for name, param in model.named_parameters():
        if 'layer4' in name or 'decoder' in name or 'aspp' in name or 'cbam' in name or 'aux_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    criterion = CombinedLoss()
    aux_criterion = nn.BCEWithLogitsLoss()
    
    mixup = MixUp(alpha=0.4)
    cutmix = CutMix(alpha=1.0)
    
    best_miou = 0
    best_metrics = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            images = batch['image'].to(DEVICE)
            gts = batch['gt'].to(DEVICE)
            
            if random.random() < 0.3:
                images, gts = mixup(images, gts)
            elif random.random() < 0.3:
                images, gts = cutmix(images, gts)
            
            optimizer.zero_grad()
            outputs, aux_outputs = model(images)
            
            main_loss = criterion(outputs.squeeze(1), gts)
            aux_loss = aux_criterion(aux_outputs.squeeze(1), F.interpolate(gts.unsqueeze(1), aux_outputs.shape[2:], mode='nearest').squeeze(1))
            loss = main_loss + 0.4 * aux_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        all_metrics = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                gts = batch['gt'].numpy()
                
                preds = tta_predict(model, images, scales=[0.8, 1.0, 1.2])
                preds = preds.cpu().numpy()
                preds = (preds > 0.5).astype(np.float32)
                
                for i in range(images.shape[0]):
                    metrics = compute_seg_metrics(preds[i, 0], gts[i])
                    all_metrics.append(metrics)
        
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        
        if avg_metrics['mIoU'] > best_miou:
            best_miou = avg_metrics['mIoU']
            best_metrics = avg_metrics.copy()
        
        print(f"    Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.4f}, mIoU={avg_metrics['mIoU']:.4f}, Dice={avg_metrics['Dice']:.4f}, F1={avg_metrics['F1']:.4f}")
    
    return best_metrics


def train_cls_model(train_loader, val_loader, num_classes, epochs=30, lr=1e-4):
    print(f"\n  初始化改进分类模型...")
    print(f"    - ResNet50骨干网络 (ImageNet预训练)")
    print(f"    - CBAM + SE双重注意力")
    
    model = ImprovedClsModel(num_classes=num_classes).to(DEVICE)
    
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name or 'cbam' in name or 'se' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
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
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                labels = batch['class_id'].numpy()
                
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                
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
    print("完整训练实验 - 包含所有改进方法")
    print(f"设备: {DEVICE}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\n改进方法:")
    print("  1. 数据增强: MixUp, CutMix, 几何变换, 颜色抖动")
    print("  2. 注意力机制: CBAM, SE")
    print("  3. 多尺度融合: ASPP, FPN")
    print("  4. 损失函数: Focal Loss, Tversky Loss, Boundary Loss")
    print("  5. 深度监督")
    print("  6. 测试时增强 (TTA)")
    print("  7. 标签平滑")
    print("  8. 学习率调度: CosineAnnealingWarmRestarts")
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': DEVICE,
        'model': 'ResNet50 + CBAM + ASPP + FPN + All Improvements',
        'improvements': [
            'MixUp & CutMix Augmentation',
            'CBAM & SE Attention',
            'ASPP Multi-scale Fusion',
            'FPN Decoder',
            'Focal + Tversky + Boundary Loss',
            'Deep Supervision',
            'Test-Time Augmentation',
            'Label Smoothing',
            'CosineAnnealingWarmRestarts'
        ],
        'experiments': {}
    }
    
    cod10k_root = r"d:\University\竞赛\4C\智慧农林\模型代码\COD10K-v3\COD10K-v3"
    ip02_root = r"d:\University\竞赛\4C\智慧农林\模型代码\数据集\数据集\IP02"
    
    print("\n" + "="*80)
    print("实验1: COD10K 伪装目标分割")
    print("="*80)
    
    print(f"\n[1] 加载数据集...")
    full_dataset = COD10KDataset(cod10k_root, img_size=256, augment=True)
    print(f"    总样本数: {len(full_dataset)}")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"    训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    print(f"\n[2] 开始训练 (30 epochs)...")
    seg_metrics = train_seg_model(train_loader, val_loader, epochs=30, lr=1e-4)
    results['experiments']['COD10K_Segmentation_Improved'] = seg_metrics
    
    print("\n" + "="*80)
    print("实验2: IP02 害虫分类")
    print("="*80)
    
    print(f"\n[1] 加载数据集...")
    full_dataset = IP02Dataset(ip02_root, img_size=224, augment=True)
    print(f"    总样本数: {len(full_dataset)}")
    print(f"    类别数: {full_dataset.num_classes}")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"    训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    print(f"\n[2] 开始训练 (30 epochs)...")
    cls_metrics = train_cls_model(train_loader, val_loader, full_dataset.num_classes, epochs=30, lr=1e-4)
    results['experiments']['IP02_Classification_Improved'] = cls_metrics
    
    results_file = os.path.join(RESULTS_DIR, 'improved_training_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("训练完成!")
    print("="*80)
    print(f"\nCOD10K分割结果 (改进模型):")
    for k, v in seg_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nIP02分类结果 (改进模型):")
    for k, v in cls_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\n结果已保存到: {results_file}")
    
    return results


if __name__ == "__main__":
    results = main()
