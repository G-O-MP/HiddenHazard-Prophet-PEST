# -*- coding: utf-8 -*-
"""
农林虫害智检系统 - 统一训练脚本

支持：
1. 小样本训练 (1-shot, 3-shot, 5-shot, 20-shot)
2. 全监督训练
3. 多数据集训练 (COD10K, IP102, CAMO)
4. 消融实验

使用方法：
    python train.py --mode fewshot --shot 5 --dataset COD10K
    python train.py --mode full --dataset IP102
    python train.py --mode ablation --ablation_id 1
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from my_models.我的模型.model_with_classification import FewShotCamouflageSegWithClass


class FewShotDataset(Dataset):
    """小样本数据集"""
    def __init__(self, root_dir, split='train', shot=5, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.shot = shot
        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.images = []
        self.masks = []
        self.labels = []
        self._load_data()
        
    def _load_data(self):
        image_dir = self.root_dir / self.split / 'Image'
        if image_dir.exists():
            for img_path in sorted(image_dir.glob('*.jpg'))[:100]:
                self.images.append(str(img_path))
                self.masks.append(str(img_path).replace('Image', 'Mask').replace('.jpg', '.png'))
                self.labels.append(0)
    
    def __len__(self):
        return max(len(self.images), 10)
    
    def __getitem__(self, idx):
        idx = idx % len(self.images) if self.images else 0
        
        if self.images:
            image = Image.open(self.images[idx]).convert('RGB')
            image = self.transform(image)
        else:
            image = torch.randn(3, 640, 640)
        
        if self.masks and os.path.exists(self.masks[idx]):
            mask = Image.open(self.masks[idx]).convert('L')
            mask = transforms.Resize((640, 640))(mask)
            mask = transforms.ToTensor()(mask)
        else:
            mask = torch.zeros(1, 640, 640)
        
        label = self.labels[idx] if self.labels else 0
        
        support_images = image.unsqueeze(0).repeat(self.shot, 1, 1, 1)
        support_masks = mask.unsqueeze(0).repeat(self.shot, 1, 1, 1)
        
        return {
            'query_image': image,
            'query_mask': mask,
            'query_label': label,
            'support_images': support_images,
            'support_masks': support_masks,
            'support_labels': torch.tensor([label] * self.shot)
        }


class FullDataset(Dataset):
    """全监督数据集"""
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.images = []
        self.masks = []
        self.labels = []
        self._load_data()
        
    def _load_data(self):
        image_dir = self.root_dir / self.split / 'Image'
        if image_dir.exists():
            for img_path in sorted(image_dir.glob('*.jpg')):
                self.images.append(str(img_path))
                self.masks.append(str(img_path).replace('Image', 'Mask').replace('.jpg', '.png'))
                self.labels.append(0)
    
    def __len__(self):
        return max(len(self.images), 10)
    
    def __getitem__(self, idx):
        idx = idx % len(self.images) if self.images else 0
        
        if self.images:
            image = Image.open(self.images[idx]).convert('RGB')
            image = self.transform(image)
        else:
            image = torch.randn(3, 640, 640)
        
        if self.masks and os.path.exists(self.masks[idx]):
            mask = Image.open(self.masks[idx]).convert('L')
            mask = transforms.Resize((640, 640))(mask)
            mask = transforms.ToTensor()(mask)
        else:
            mask = torch.zeros(1, 640, 640)
        
        label = self.labels[idx] if self.labels else 0
        
        return {
            'image': image,
            'mask': mask,
            'label': label
        }


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        seg_loss = self.bce(outputs['seg_logits'], targets['mask']) + \
                   self.dice(outputs['seg_logits'], targets['mask'])
        
        cls_loss = self.ce(outputs['cls_logits'], targets['label'])
        
        box_loss = F.smooth_l1_loss(outputs['bbox'], targets['bbox'])
        
        return seg_loss + 0.5 * cls_loss + 0.3 * box_loss


def train_fewshot(args):
    """小样本训练"""
    print(f"\n{'='*60}")
    print(f"小样本训练 - {args.shot}-shot")
    print(f"数据集: {args.dataset}")
    print(f"{'='*60}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = FewShotCamouflageSegWithClass(
        backbone='resnet50',
        num_classes=102,
        use_adapter=True
    ).to(device)
    
    if args.resume:
        print(f"加载预训练权重: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device), strict=False)
    
    train_dataset = FewShotDataset(args.data_root, split='Train', shot=args.shot)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    save_dir = Path(args.output_dir) / f'fewshot_{args.shot}shot_{args.dataset}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            query_image = batch['query_image'].to(device)
            query_mask = batch['query_mask'].to(device)
            query_label = batch['query_label'].to(device)
            support_images = batch['support_images'].to(device)
            support_masks = batch['support_masks'].to(device)
            support_labels = batch['support_labels'].to(device)
            
            outputs = model(query_image, support_images, support_masks)
            
            h, w = query_mask.shape[-2:]
            bbox = torch.tensor([0.5, 0.5, 0.3, 0.3], device=device).unsqueeze(0).repeat(query_image.size(0), 1)
            
            targets = {
                'mask': query_mask,
                'label': query_label,
                'bbox': bbox
            }
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
        
        if (epoch + 1) % args.save_interval == 0:
            save_path = save_dir / f'model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f'模型已保存: {save_path}')
    
    final_path = save_dir / 'model_final.pth'
    torch.save(model.state_dict(), final_path)
    print(f'\n训练完成！最终模型已保存: {final_path}')
    
    return model


def train_full(args):
    """全监督训练"""
    print(f"\n{'='*60}")
    print(f"全监督训练")
    print(f"数据集: {args.dataset}")
    print(f"{'='*60}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = FewShotCamouflageSegWithClass(
        backbone='resnet50',
        num_classes=102,
        use_adapter=True
    ).to(device)
    
    train_dataset = FullDataset(args.data_root, split='Train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    save_dir = Path(args.output_dir) / f'full_{args.dataset}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            label = batch['label'].to(device)
            
            support_images = image.unsqueeze(1).repeat(1, 5, 1, 1, 1)
            support_masks = mask.unsqueeze(1).repeat(1, 5, 1, 1, 1)
            
            outputs = model(image, support_images, support_masks)
            
            bbox = torch.tensor([0.5, 0.5, 0.3, 0.3], device=device).unsqueeze(0).repeat(image.size(0), 1)
            
            targets = {
                'mask': mask,
                'label': label,
                'bbox': bbox
            }
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
        
        if (epoch + 1) % args.save_interval == 0:
            save_path = save_dir / f'model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f'模型已保存: {save_path}')
    
    final_path = save_dir / 'model_final.pth'
    torch.save(model.state_dict(), final_path)
    print(f'\n训练完成！最终模型已保存: {final_path}')
    
    return model


def train_ablation(args):
    """消融实验训练"""
    print(f"\n{'='*60}")
    print(f"消融实验 - ID: {args.ablation_id}")
    print(f"{'='*60}\n")
    
    ablation_configs = {
        1: {'use_adapter': False, 'use_tta': False, 'use_attention': False},
        2: {'use_adapter': True, 'use_tta': False, 'use_attention': False},
        3: {'use_adapter': True, 'use_tta': True, 'use_attention': False},
        4: {'use_adapter': True, 'use_tta': True, 'use_attention': True},
        5: {'use_adapter': True, 'use_tta': True, 'use_attention': True, 'use_proto_refine': True},
        6: {'use_adapter': True, 'use_tta': True, 'use_attention': True, 'use_proto_refine': True, 'use_gnn': True},
    }
    
    config = ablation_configs.get(args.ablation_id, ablation_configs[1])
    print(f"消融配置: {config}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FewShotCamouflageSegWithClass(
        backbone='resnet50',
        num_classes=102,
        **config
    ).to(device)
    
    train_dataset = FewShotDataset(args.data_root, split='Train', shot=5)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    save_dir = Path(args.output_dir) / f'ablation_{args.ablation_id}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            query_image = batch['query_image'].to(device)
            query_mask = batch['query_mask'].to(device)
            query_label = batch['query_label'].to(device)
            support_images = batch['support_images'].to(device)
            support_masks = batch['support_masks'].to(device)
            
            outputs = model(query_image, support_images, support_masks)
            
            bbox = torch.tensor([0.5, 0.5, 0.3, 0.3], device=device).unsqueeze(0).repeat(query_image.size(0), 1)
            targets = {'mask': query_mask, 'label': query_label, 'bbox': bbox}
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
    
    final_path = save_dir / 'model_final.pth'
    torch.save(model.state_dict(), final_path)
    print(f'\n消融实验 {args.ablation_id} 完成！模型已保存: {final_path}')
    
    return model


def main():
    parser = argparse.ArgumentParser(description='农林虫害智检系统 - 训练脚本')
    
    parser.add_argument('--mode', type=str, default='fewshot', 
                       choices=['fewshot', 'full', 'ablation'],
                       help='训练模式')
    parser.add_argument('--dataset', type=str, default='COD10K',
                       choices=['COD10K', 'IP102', 'CAMO', 'NC4K'],
                       help='数据集名称')
    parser.add_argument('--shot', type=int, default=5,
                       choices=[1, 3, 5, 20],
                       help='小样本设置')
    parser.add_argument('--ablation_id', type=int, default=1,
                       choices=[1, 2, 3, 4, 5, 6],
                       help='消融实验ID')
    
    parser.add_argument('--data_root', type=str, 
                       default='d:/University/竞赛/4C/智慧农林/模型代码/COD10K-v3/COD10K-v3',
                       help='数据集根目录')
    parser.add_argument('--output_dir', type=str, 
                       default='d:/University/竞赛/4C/智慧农林/模型代码/checkpoints',
                       help='模型保存目录')
    parser.add_argument('--resume', type=str, default='',
                       help='预训练权重路径')
    
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='保存间隔')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"农林虫害智检系统 - 训练脚本")
    print(f"{'='*60}")
    print(f"模式: {args.mode}")
    print(f"数据集: {args.dataset}")
    if args.mode == 'fewshot':
        print(f"Shot: {args.shot}")
    elif args.mode == 'ablation':
        print(f"消融ID: {args.ablation_id}")
    print(f"{'='*60}\n")
    
    if args.mode == 'fewshot':
        train_fewshot(args)
    elif args.mode == 'full':
        train_full(args)
    elif args.mode == 'ablation':
        train_ablation(args)


if __name__ == '__main__':
    main()
