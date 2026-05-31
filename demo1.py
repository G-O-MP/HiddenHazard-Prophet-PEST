# -*- coding: utf-8 -*-
"""
改进版小样本伪装目标分割训练脚本 (demo1.py)

8项改进:
1. 注意力机制: CBAM + SE + ECA 三重注意力
2. 多尺度融合: ASPP + FPN
3. 数据增强: MixUp, CutMix, 几何变换, 颜色抖动
4. 损失函数: Focal Loss + Tversky Loss + Boundary Loss
5. 深度监督
6. 测试时增强 (TTA)
7. 标签平滑
8. 学习率调度: CosineAnnealingWarmRestarts
"""
import os
import sys
import random
import math
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.metrics_cod import compute_all_metrics
from my_models.fewshot_camouflage_seg import create_fscamo_model, freeze_encoder_bn


# =======================
# 配置
# =======================
# 数据路径: splits和图像都在当前工作目录的FSL-COD数据集下
DATA_ROOT = os.path.join(PROJECT_ROOT, "FSL-COD数据集")
SPLIT_DIR = os.path.join(DATA_ROOT, "splits")

BASE_TRAIN_LIST = os.path.join(SPLIT_DIR, "base_train.txt")
BASE_VAL_LIST = os.path.join(SPLIT_DIR, "base_val.txt")
NOVEL_SUPPORT_LIST = os.path.join(SPLIT_DIR, "novel_support.txt")
NOVEL_QUERY_LIST = os.path.join(SPLIT_DIR, "novel_query.txt")

IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

N_WAY = 1
K_SHOT = 5
N_QUERY = 1
PATIENCE = 10


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =======================
# 数据增强: MixUp, CutMix, 几何变换, 颜色抖动
# =======================

class EnhancedTransform:
    """增强的数据变换: MixUp, CutMix, 几何变换, 颜色抖动"""
    
    def __init__(self, img_size=512, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        
        # 颜色抖动
        if is_train:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
    
    def __call__(self, img, mask):
        # 调整大小
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        if self.is_train:
            # 几何变换: 随机翻转
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 几何变换: 随机旋转 (±10°)
            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                img = img.rotate(angle, resample=Image.BILINEAR)
                mask = mask.rotate(angle, resample=Image.NEAREST)
            
            # 颜色抖动
            img = self.color_jitter(img)
        
        # 转换为tensor
        img = self.to_tensor(img)
        img = self.normalize(img)
        
        # 处理mask
        mask = np.array(mask, dtype=np.float32)
        if mask.max() > 1:
            mask = (mask > 127.5).astype(np.float32)
        else:
            mask = (mask > 0.5).astype(np.float32)
        
        mask = torch.from_numpy(mask)
        return img, mask
    
    def mixup(self, img1, mask1, img2, mask2, alpha=0.2):
        """
        MixUp数据增强
        img = λ*img1 + (1-λ)*img2
        mask = λ*mask1 + (1-λ)*mask2
        """
        lam = np.random.beta(alpha, alpha)
        img = lam * img1 + (1 - lam) * img2
        mask = lam * mask1 + (1 - lam) * mask2
        return img, mask
    
    def cutmix(self, img1, mask1, img2, mask2, beta=1.0):
        """
        CutMix数据增强
        从img2裁剪patch覆盖img1,按面积比例混合标签
        """
        lam = np.random.beta(beta, beta)
        
        # 计算裁剪区域
        H, W = img1.shape[1], img1.shape[2]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # 随机中心点
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # 边界限制
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # 执行CutMix
        img = img1.clone()
        mask = mask1.clone()
        img[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]
        mask[bby1:bby2, bbx1:bbx2] = mask2[bby1:bby2, bbx1:bbx2]
        
        # 计算实际的lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
        
        return img, mask, lam


# =======================
# 标签平滑
# =======================

def label_smoothing(mask, epsilon=0.1):
    """
    标签平滑: 将硬标签{0,1}转换为软标签{ε, 1-ε}
    
    Args:
        mask: (B, 1, H, W) 硬标签
        epsilon: 平滑系数
    
    Returns:
        smooth_mask: 软标签
    """
    return mask * (1 - epsilon) + (1 - mask) * epsilon


# =======================
# 损失函数: Focal + Tversky + Boundary
# =======================

class FocalLoss(nn.Module):
    """
    Focal Loss: 降低易分类样本权重,关注难样本
    FL = -α*(1-p)^γ*log(p)
    """
    def __init__(self, gamma=2.0, alpha=0.75, eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
    
    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        prob = prob.clamp(self.eps, 1 - self.eps)
        
        # 交叉熵
        ce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        
        # focal权重
        pt = prob * target + (1 - prob) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma
        
        loss = self.alpha * focal_weight * ce_loss
        return loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss: 不平衡分割优化
    TL = 1 - TP/(TP+α*FN+β*FP)
    α控制假阴权重,β控制假阳权重
    """
    def __init__(self, alpha=0.7, beta=0.3, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    
    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        
        # 真阳性、假阴性、假阳性
        TP = (prob * target).sum(dim=(2, 3))
        FN = ((1 - prob) * target).sum(dim=(2, 3))
        FP = (prob * (1 - target)).sum(dim=(2, 3))
        
        # Tversky指数
        tversky_index = (TP + self.eps) / (TP + self.alpha * FN + self.beta * FP + self.eps)
        loss = 1 - tversky_index
        
        return loss.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss: 基于距离图的边界对齐损失
    使用形态学操作生成边界区域
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
    
    def make_edge(self, mask):
        """生成边界mask"""
        pad = self.kernel_size // 2
        dil = F.max_pool2d(mask, kernel_size=self.kernel_size, stride=1, padding=pad)
        ero = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=self.kernel_size, stride=1, padding=pad)
        edge = (dil - ero).clamp(0, 1)
        return edge
    
    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        
        # 生成预测和真实的边界
        edge_pred = self.make_edge(prob)
        edge_gt = self.make_edge(target)
        
        # BCE on boundary
        loss = F.binary_cross_entropy(edge_pred.clamp(1e-6, 1-1e-6), edge_gt)
        
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失: Focal + Tversky + Boundary
    支持多尺度深度监督
    """
    def __init__(self, focal_weight=1.0, tversky_weight=1.0, boundary_weight=0.5,
                 scale_weights=None):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.75)
        self.tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
        self.boundary_loss = BoundaryLoss(kernel_size=7)
        
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        
        # 多尺度权重
        if scale_weights is None:
            self.scale_weights = {'log2': 1.0, 'log3': 0.5, 'log4': 0.25, 'log5': 0.125}
        else:
            self.scale_weights = scale_weights
    
    def forward(self, logits, target):
        """单尺度损失"""
        loss_focal = self.focal_loss(logits, target)
        loss_tversky = self.tversky_loss(logits, target)
        loss_boundary = self.boundary_loss(logits, target)
        
        total = (self.focal_weight * loss_focal + 
                self.tversky_weight * loss_tversky + 
                self.boundary_weight * loss_boundary)
        
        return total
    
    def forward_multiscale(self, logits_dict, target_mask):
        """多尺度损失(深度监督)"""
        total_loss = 0.0
        
        for lvl, weight in self.scale_weights.items():
            if lvl in logits_dict:
                logit = logits_dict[lvl]
                # 下采样target到对应尺度
                target = F.interpolate(
                    target_mask, 
                    size=logit.shape[-2:], 
                    mode='nearest'
                )
                
                # 标签平滑
                smooth_target = label_smoothing(target, epsilon=0.1)
                
                loss = self.forward(logit, smooth_target)
                total_loss += weight * loss
        
        return total_loss


# =======================
# 数据工具
# =======================

def load_split_list(txt_path):
    """读取txt文件,返回list[(img, mask, scene)]"""
    items = []
    
    # 旧路径前缀(在txt文件中)
    old_prefix = r"D:\University\Homework\机器学习\机器学习课程设计\FSL-COD数据集"
    # 新路径前缀(实际数据位置)
    new_prefix = DATA_ROOT
    
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                img, mask, scene = parts[0], parts[1], parts[2]
                # 替换路径前缀
                img = img.replace(old_prefix, new_prefix)
                mask = mask.replace(old_prefix, new_prefix)
            else:
                img, mask, scene = parts[0], parts[1], "unknown"
                img = img.replace(old_prefix, new_prefix)
                mask = mask.replace(old_prefix, new_prefix)
            items.append((img, mask, scene))
    return items


class FewShotEpisodeDataset(Dataset):
    """Episode采样数据集(支持MixUp/CutMix)"""
    
    def __init__(self, split_list, img_size=512, k_shot=5, n_query=1, use_mixup=False, use_cutmix=False):
        super().__init__()
        self.k_shot = k_shot
        self.n_query = n_query
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.transform = EnhancedTransform(img_size=img_size, is_train=True)

        self.by_scene = defaultdict(list)
        for img, mask, scene in split_list:
            self.by_scene[scene].append((img, mask))

        self.scenes = list(self.by_scene.keys())

    def __len__(self):
        return 100000

    def _load_img_mask(self, img_path, mask_path):
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return self.transform(img, mask)

    def __getitem__(self, idx):
        scene = random.choice(self.scenes)
        pairs = self.by_scene[scene]

        min_required = self.k_shot + self.n_query
        if len(pairs) < min_required:
            pairs = pairs * (min_required // len(pairs) + 1)

        idxs = list(range(len(pairs)))
        random.shuffle(idxs)
        support_idx = idxs[:self.k_shot]
        query_idx = idxs[self.k_shot:self.k_shot + self.n_query]

        support_imgs, support_masks = [], []
        query_imgs, query_masks = [], []

        for i in support_idx:
            img_path, mask_path = pairs[i]
            img, mask = self._load_img_mask(img_path, mask_path)
            support_imgs.append(img)
            support_masks.append(mask)

        for i in query_idx:
            img_path, mask_path = pairs[i]
            img, mask = self._load_img_mask(img_path, mask_path)
            query_imgs.append(img)
            query_masks.append(mask)

        # MixUp/CutMix增强 (仅在support内或query内)
        if self.use_mixup and len(support_imgs) > 1:
            if random.random() < 0.5:
                idx1, idx2 = random.sample(range(len(support_imgs)), 2)
                support_imgs[idx1], support_masks[idx1] = self.transform.mixup(
                    support_imgs[idx1], support_masks[idx1],
                    support_imgs[idx2], support_masks[idx2]
                )

        if self.use_cutmix and len(support_imgs) > 1:
            if random.random() < 0.5:
                idx1, idx2 = random.sample(range(len(support_imgs)), 2)
                img_new, mask_new, _ = self.transform.cutmix(
                    support_imgs[idx1], support_masks[idx1],
                    support_imgs[idx2], support_masks[idx2]
                )
                support_imgs[idx1] = img_new
                support_masks[idx1] = mask_new

        support_imgs = torch.stack(support_imgs, dim=0)
        support_masks = torch.stack(support_masks, dim=0).unsqueeze(1)  # (K, 1, H, W)
        query_imgs = torch.stack(query_imgs, dim=0)
        query_masks = torch.stack(query_masks, dim=0).unsqueeze(1)  # (N, 1, H, W)

        return support_imgs, support_masks, query_imgs, query_masks, scene


# =======================
# 评估函数
# =======================

@torch.no_grad()
def evaluate_on_val(model, val_items, support_items, device, use_tta=False):
    """在验证集上评估(支持TTA)"""
    model.eval()
    
    # 按场景分组support
    support_by_scene = defaultdict(list)
    for img, mask, scene in support_items:
        support_by_scene[scene].append((img, mask))
    
    transform = EnhancedTransform(img_size=IMG_SIZE, is_train=False)
    
    all_preds = []
    all_gts = []
    
    for img_path, mask_path, scene in val_items:
        # 加载query
        qry_img = Image.open(img_path).convert("RGB")
        qry_mask = Image.open(mask_path).convert("L")
        qry_img, qry_mask = transform(qry_img, qry_mask)
        qry_img = qry_img.unsqueeze(0).to(device)
        qry_mask_np = qry_mask.numpy()
        
        # 找到同场景的support
        if scene in support_by_scene:
            supp_pairs = support_by_scene[scene][:K_SHOT]
        else:
            rand_scene = random.choice(list(support_by_scene.keys()))
            supp_pairs = support_by_scene[rand_scene][:K_SHOT]
        
        # 加载support
        supp_imgs, supp_masks = [], []
        for sp_img, sp_mask in supp_pairs:
            s_img = Image.open(sp_img).convert("RGB")
            s_mask = Image.open(sp_mask).convert("L")
            s_img, s_mask = transform(s_img, s_mask)
            supp_imgs.append(s_img)
            supp_masks.append(s_mask)
        
        # 填充至K_SHOT
        while len(supp_imgs) < K_SHOT:
            supp_imgs.append(supp_imgs[0])
            supp_masks.append(supp_masks[0])
        
        supp_imgs = torch.stack(supp_imgs).to(device)
        supp_masks = torch.stack(supp_masks).unsqueeze(1).to(device)  # (K, 1, H, W)
        
        if use_tta:
            # TTA: 多尺度 + 翻转
            preds_multi = []
            scales = [0.9, 1.0, 1.1]
            
            for scale in scales:
                # 原始尺度
                if scale != 1.0:
                    new_h, new_w = int(IMG_SIZE * scale), int(IMG_SIZE * scale)
                    supp_imgs_s = F.interpolate(supp_imgs, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    supp_masks_s = F.interpolate(supp_masks, size=(new_h, new_w), mode='nearest')
                    qry_img_s = F.interpolate(qry_img, size=(new_h, new_w), mode='bilinear', align_corners=False)
                else:
                    supp_imgs_s, supp_masks_s, qry_img_s = supp_imgs, supp_masks, qry_img
                
                out, _, _ = model(supp_imgs_s, supp_masks_s, qry_img_s)
                pred = F.interpolate(torch.sigmoid(out), size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
                preds_multi.append(pred)
                
                # 水平翻转
                supp_imgs_flip = torch.flip(supp_imgs_s, dims=[3])
                supp_masks_flip = torch.flip(supp_masks_s, dims=[3])
                qry_img_flip = torch.flip(qry_img_s, dims=[3])
                
                out_flip, _, _ = model(supp_imgs_flip, supp_masks_flip, qry_img_flip)
                pred_flip = torch.flip(torch.sigmoid(out_flip), dims=[3])
                pred_flip = F.interpolate(pred_flip, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
                preds_multi.append(pred_flip)
            
            # 平均融合
            final_pred = torch.stack(preds_multi).mean(dim=0)
            pred_prob = final_pred.cpu().numpy()[0, 0]
        else:
            # 正常推理
            out, _, _ = model(supp_imgs, supp_masks, qry_img)
            pred_prob = torch.sigmoid(out).cpu().numpy()[0, 0]  # (H, W)
        
        all_preds.append(pred_prob)
        all_gts.append(qry_mask_np)
    
    all_preds = np.stack(all_preds, axis=0)
    all_gts = np.stack(all_gts, axis=0)
    
    metrics = compute_all_metrics(all_preds, all_gts)
    return metrics


# =======================
# 训练函数
# =======================

def train_demo1(num_epochs=30, steps_per_epoch=200, lr=1e-4, use_tta_eval=False):
    """
    改进版训练流程
    
    Args:
        num_epochs: 训练轮数
        steps_per_epoch: 每轮步数
        lr: 初始学习率
        use_tta_eval: 评估时是否使用TTA
    """
    print("="*60)
    print("改进版Few-shot Camouflage Segmentation训练 (demo1)")
    print("="*60)
    print("\n8项改进:")
    print("1. ✓ 三重注意力 (CBAM + SE + ECA)")
    print("2. ✓ 多尺度融合 (ASPP + FPN)")
    print("3. ✓ 数据增强 (MixUp + CutMix + 几何 + 颜色)")
    print("4. ✓ 组合损失 (Focal + Tversky + Boundary)")
    print("5. ✓ 深度监督 (多尺度损失)")
    print("6. ✓ 测试时增强 (TTA)")
    print("7. ✓ 标签平滑 (epsilon=0.1)")
    print("8. ✓ 学习率调度 (CosineAnnealingWarmRestarts)")
    print("="*60)
    
    base_train_items = load_split_list(BASE_TRAIN_LIST)
    base_val_items = load_split_list(BASE_VAL_LIST)

    # 创建数据集(启用MixUp和CutMix)
    train_dataset = FewShotEpisodeDataset(
        base_train_items, img_size=IMG_SIZE, k_shot=K_SHOT, n_query=N_QUERY,
        use_mixup=True, use_cutmix=True
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # 创建模型(启用FPN和注意力)
    model = create_fscamo_model(
        backbone='resnet50',
        pretrained=True,
        use_edge=True,
        use_cross_attn=True,
        use_token_proto=False,
        use_adapter=True,
        adapt_levels=("p2", "p3"),
        freeze_encoder_batchnorm=True,
        use_fpn=True,
        use_attention=True
    ).to(DEVICE)
    
    # 损失函数(组合损失)
    criterion = CombinedLoss(
        focal_weight=1.0,
        tversky_weight=1.0,
        boundary_weight=0.5
    )
    
    # 优化器(AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度器(CosineAnnealingWarmRestarts)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_dice = 0.0
    no_improve = 0
    
    save_name = "demo1_improved_model.pth"
    save_path = os.path.join("checkpoints", save_name)
    os.makedirs("checkpoints", exist_ok=True)

    print(f"\n设备: {DEVICE}")
    print(f"K-shot: {K_SHOT}")
    print(f"训练集场景数: {len(set(x[2] for x in base_train_items))}")
    print(f"验证集样本数: {len(base_val_items)}")
    print(f"早停patience: {PATIENCE}")
    print(f"TTA评估: {'启用' if use_tta_eval else '禁用'}")
    print("="*60)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
                
            supp_imgs, supp_masks, qry_imgs, qry_masks, scene = batch
            
            # 去掉batch维度
            supp_imgs = supp_imgs.squeeze(0).to(DEVICE)
            supp_masks = supp_masks.squeeze(0).to(DEVICE)
            qry_imgs = qry_imgs.squeeze(0).to(DEVICE)
            qry_masks = qry_masks.squeeze(0).to(DEVICE)

            optimizer.zero_grad()
            
            # 前向传播
            out, logits, guides = model(supp_imgs, supp_masks, qry_imgs)
            
            # 计算多尺度损失(深度监督)
            loss = criterion.forward_multiscale(logits, qry_masks)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 更新学习率
        scheduler.step(epoch)
        avg_loss = running_loss / steps_per_epoch

        # 验证
        metrics = evaluate_on_val(model, base_val_items, base_train_items, DEVICE, use_tta=use_tta_eval)
        val_dice = metrics['Dice']

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"LR: {current_lr:.6f} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | "
              f"mIoU: {metrics['mIoU']:.4f} | "
              f"S: {metrics['S_measure']:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> 新最佳! Dice={best_val_dice:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n早停触发! 连续{PATIENCE}轮无改进")
                break

    print(f"\n[训练完成] 最佳Dice: {best_val_dice:.4f}")
    print(f"模型保存至: {save_path}")
    return save_path


# =======================
# Novel评估
# =======================

@torch.no_grad()
def evaluate_novel(model_path=None, use_tta=False):
    """Novel场景评估"""
    print("\n" + "="*60)
    print(f"Novel评估 (TTA={'启用' if use_tta else '禁用'})")
    print("="*60)
    
    novel_support_items = load_split_list(NOVEL_SUPPORT_LIST)
    novel_query_items = load_split_list(NOVEL_QUERY_LIST)
    
    # 创建模型
    model = create_fscamo_model(
        backbone='resnet50',
        pretrained=False,
        use_edge=True,
        use_cross_attn=True,
        use_token_proto=False,
        use_adapter=True,
        adapt_levels=("p2", "p3"),
        freeze_encoder_batchnorm=True,
        use_fpn=True,
        use_attention=True
    ).to(DEVICE)
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"已加载模型: {model_path}")
    else:
        print("警告: 未找到模型,使用随机初始化")
    
    print(f"Novel Support: {len(novel_support_items)} 样本")
    print(f"Novel Query: {len(novel_query_items)} 样本")
    
    metrics = evaluate_on_val(model, novel_query_items, novel_support_items, DEVICE, use_tta=use_tta)
    
    print("\n" + "="*60)
    print("Novel测试结果:")
    print("="*60)
    print("主指标:")
    print(f"  mIoU:      {metrics['mIoU']:.4f}")
    print(f"  Dice:      {metrics['Dice']:.4f}")
    print(f"  S-measure: {metrics['S_measure']:.4f}")
    print("辅指标:")
    print(f"  maxF:      {metrics['maxF']:.4f}")
    print(f"  MAE:       {metrics['MAE']:.4f}")
    print(f"  PixelAcc:  {metrics['PixelAcc']:.4f}")
    print("="*60)
    
    return metrics


# =======================
# 主函数
# =======================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='改进版小样本伪装目标分割训练')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--steps', type=int, default=200, help='每轮步数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--eval-only', action='store_true', help='仅评估')
    parser.add_argument('--use-tta', action='store_true', help='使用TTA评估')
    parser.add_argument('--model-path', type=str, default=None, help='模型路径')
    
    args = parser.parse_args()
    
    print("="*60)
    print("改进版Few-shot Camouflage Segmentation (demo1)")
    print("="*60)
    print(f"设备: {DEVICE}")
    print(f"TTA: {'启用' if args.use_tta else '禁用'}")
    
    if not args.eval_only:
        # 训练
        model_path = train_demo1(
            num_epochs=args.epochs,
            steps_per_epoch=args.steps,
            lr=args.lr,
            use_tta_eval=args.use_tta
        )
    else:
        model_path = args.model_path
    
    # Novel评估
    evaluate_novel(model_path=model_path, use_tta=args.use_tta)


if __name__ == "__main__":
    main()
