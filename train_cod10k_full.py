# -*- coding: utf-8 -*-
"""
COD10K-v3 全监督伪装目标分割训练脚本

使用改进模型(FPN+ASPP+三重注意力)在COD10K数据集上进行全监督训练
"""
import os
import sys
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.metrics_cod import compute_all_metrics
from my_models.fewshot_camouflage_seg import create_fscamo_model, freeze_encoder_bn
from demo1 import CombinedLoss, EnhancedTransform


# =======================
# 配置
# =======================
DATA_ROOT = os.path.join(PROJECT_ROOT, "COD10K-v3", "COD10K-v3")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "Train", "Image")
TRAIN_GT_DIR = os.path.join(DATA_ROOT, "Train", "GT_Object")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "Test", "Image")
TEST_GT_DIR = os.path.join(DATA_ROOT, "Test", "GT_Object")

IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BATCH_SIZE = 8  # 全监督可以使用更大的batch size
NUM_WORKERS = 4
PATIENCE = 10


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =======================
# COD10K数据集
# =======================

class COD10KDataset(Dataset):
    """COD10K全监督数据集"""
    
    def __init__(self, img_dir, gt_dir, img_size=512, is_train=True):
        super().__init__()
        self.img_size = img_size
        self.is_train = is_train
        self.transform = EnhancedTransform(img_size=img_size, is_train=is_train)
        
        # 获取所有图像文件
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.gt_files = []
        
        for img_file in self.img_files:
            img_name = os.path.basename(img_file).replace('.jpg', '.png')
            gt_file = os.path.join(gt_dir, img_name)
            if os.path.exists(gt_file):
                self.gt_files.append(gt_file)
            else:
                # 有些可能是.jpg格式
                gt_file_jpg = os.path.join(gt_dir, img_name.replace('.png', '.jpg'))
                if os.path.exists(gt_file_jpg):
                    self.gt_files.append(gt_file_jpg)
                else:
                    print(f"Warning: No GT found for {img_name}")
        
        print(f"Loaded {len(self.img_files)} images from {img_dir}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        gt_path = self.gt_files[idx]
        
        # 加载图像和GT
        img = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")
        
        # 应用变换
        img, gt = self.transform(img, gt)
        
        return img, gt.unsqueeze(0)  # (3, H, W), (1, H, W)


# =======================
# 评估函数
# =======================

@torch.no_grad()
def evaluate_cod10k(model, test_dataset, device):
    """在COD10K测试集上评估"""
    model.eval()
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    all_preds = []
    all_gts = []
    
    for imgs, gts in test_loader:
        imgs = imgs.to(device)
        gts = gts.to(device)
        
        # 全监督模式: 直接输入图像
        # 需要修改模型forward以支持全监督
        preds = model.forward_full_supervised(imgs)
        preds = F.interpolate(preds, size=imgs.shape[2:], mode='bilinear', align_corners=False)
        pred_probs = torch.sigmoid(preds).cpu().numpy()
        
        all_preds.append(pred_probs)
        all_gts.append(gts.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    
    metrics = compute_all_metrics(all_preds, all_gts)
    return metrics


# =======================
# 训练函数
# =======================

def train_cod10k_full_supervised(num_epochs=50, lr=1e-4):
    """
    COD10K全监督训练
    
    Args:
        num_epochs: 训练轮数
        lr: 初始学习率
    """
    print("="*60)
    print("COD10K-v3 全监督训练 (改进模型)")
    print("="*60)
    print("\n数据集:")
    print(f"  训练集: {len(glob.glob(os.path.join(TRAIN_IMG_DIR, '*.jpg')))} 张")
    print(f"  测试集: {len(glob.glob(os.path.join(TEST_IMG_DIR, '*.jpg')))} 张")
    print("\n模型改进:")
    print("1. ✓ 三重注意力 (CBAM + SE + ECA)")
    print("2. ✓ 多尺度融合 (ASPP + FPN)")
    print("3. ✓ 数据增强 (MixUp + CutMix + 几何 + 颜色)")
    print("4. ✓ 组合损失 (Focal + Tversky + Boundary)")
    print("5. ✓ 深度监督 (多尺度损失)")
    print("6. ✓ 标签平滑 (epsilon=0.1)")
    print("7. ✓ 学习率调度 (CosineAnnealingWarmRestarts)")
    print("="*60)
    
    # 创建数据集
    train_dataset = COD10KDataset(TRAIN_IMG_DIR, TRAIN_GT_DIR, IMG_SIZE, is_train=True)
    test_dataset = COD10KDataset(TEST_IMG_DIR, TEST_GT_DIR, IMG_SIZE, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    # 创建模型(全监督模式)
    model = create_fscamo_model(
        backbone='resnet50',
        pretrained=True,
        use_edge=True,
        use_cross_attn=False,  # 全监督不需要cross-attention
        use_token_proto=False,
        use_adapter=False,
        adapt_levels=(),
        freeze_encoder_batchnorm=True,
        use_fpn=True,
        use_attention=True
    ).to(DEVICE)
    
    # 损失函数
    criterion = CombinedLoss(
        focal_weight=1.0,
        tversky_weight=1.0,
        boundary_weight=0.5
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_val_dice = 0.0
    no_improve = 0
    
    save_path = os.path.join("checkpoints", "cod10k_full_supervised.pth")
    os.makedirs("checkpoints", exist_ok=True)
    
    print(f"\n设备: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    print(f"每Epoch步数: {len(train_loader)}")
    print(f"早停patience: {PATIENCE}")
    print("="*60)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for step, (imgs, gts) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            gts = gts.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 全监督前向传播
            out, logits, guides = model.forward_full_supervised_with_logits(imgs)
            
            # 计算多尺度损失
            loss = criterion.forward_multiscale(logits, gts)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (step + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}] Step [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        scheduler.step(epoch)
        avg_loss = running_loss / len(train_loader)
        
        # 验证
        if (epoch + 1) % 5 == 0:  # 每5个epoch评估一次
            metrics = evaluate_cod10k(model, test_dataset, DEVICE)
            val_dice = metrics['Dice']
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch [{epoch+1:2d}/{num_epochs}] "
                  f"LR: {current_lr:.6f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
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
                if no_improve >= PATIENCE * 5:  # 因为每5个epoch才评估
                    print(f"\n早停触发! 连续{PATIENCE}次评估无改进")
                    break
        else:
            print(f"Epoch [{epoch+1:2d}/{num_epochs}] Avg Loss: {avg_loss:.4f}")
    
    print(f"\n[训练完成] 最佳Dice: {best_val_dice:.4f}")
    print(f"模型保存至: {save_path}")
    
    # 最终评估
    print("\n" + "="*60)
    print("最终测试结果")
    print("="*60)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    metrics = evaluate_cod10k(model, test_dataset, DEVICE)
    
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
    parser = argparse.ArgumentParser(description='COD10K全监督训练')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    
    args = parser.parse_args()
    
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    
    train_cod10k_full_supervised(num_epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
