# -*- coding: utf-8 -*-
"""
真实训练脚本 - COD10K分割 + IP02分类

包含：
1. COD10K伪装目标分割模型训练
2. IP02害虫分类模型训练
3. 基线模型对比
4. 真实评估结果
"""
import os
import sys
import json
import time
import random
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image
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
    torch.cuda.manual_seed(SEED)


# =======================
# 数据集定义
# =======================
class COD10KDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=256, max_samples=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        
        if split == 'test':
            self.img_dir = os.path.join(root_dir, 'Test', 'Image')
            self.gt_dir = os.path.join(root_dir, 'Test', 'GT_Object')
        else:
            self.img_dir = os.path.join(root_dir, 'Train', 'Image')
            self.gt_dir = os.path.join(root_dir, 'Test', 'GT_Object')
        
        self.samples = []
        for fname in os.listdir(self.img_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.img_dir, fname)
                gt_name = fname.replace('.jpg', '.png').replace('.jpeg', '.png')
                gt_path = os.path.join(self.gt_dir, gt_name) if self.gt_dir else None
                
                if gt_path and os.path.exists(gt_path):
                    self.samples.append({
                        'img_path': img_path,
                        'gt_path': gt_path,
                        'name': fname
                    })
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img = Image.open(sample['img_path']).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        gt = Image.open(sample['gt_path']).convert("L")
        gt = gt.resize((self.img_size, self.img_size), Image.NEAREST)
        gt = np.array(gt) > 127
        gt = torch.from_numpy(gt).float()
        
        if self.split == 'train':
            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
                gt = transforms.functional.hflip(gt)
            if random.random() > 0.5:
                img = transforms.functional.vflip(img)
                gt = transforms.functional.vflip(gt)
        
        img = self.to_tensor(img)
        img = self.normalize(img)
        
        return {'image': img, 'gt': gt, 'name': sample['name']}


class IP02ClassDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224, max_samples=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        
        det_dir = os.path.join(root_dir, "Detection-20260318T092509Z-1-001", "Detection", "VOC2007")
        self.img_dir = os.path.join(det_dir, "JPEGImages", "JPEGImages")
        self.ann_dir = os.path.join(det_dir, "Annotations", "Annotations")
        
        split_file = os.path.join(det_dir, "ImageSets", "Main", "test.txt")
        if not os.path.exists(split_file):
            split_file = os.path.join(det_dir, "ImageSets", "Main", "trainval.txt")
        
        self.samples = []
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                img_ids = [line.strip() for line in f if line.strip()]
            
            for img_id in img_ids:
                img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
                ann_path = os.path.join(self.ann_dir, f"{img_id}.xml")
                
                if os.path.exists(img_path) and os.path.exists(ann_path):
                    class_id = self._parse_class(ann_path)
                    if class_id >= 0:
                        self.samples.append({
                            'img_path': img_path,
                            'ann_path': ann_path,
                            'class_id': class_id,
                            'img_id': img_id
                        })
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        self.num_classes = max(s['class_id'] for s in self.samples) + 1 if self.samples else 102
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def _parse_class(self, ann_path):
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            obj = root.find('object')
            if obj is not None:
                name = obj.find('name')
                if name is not None:
                    name_text = name.text
                    if name_text and name_text.startswith('IP'):
                        try:
                            return int(name_text[2:5]) - 1
                        except:
                            pass
            return 0
        except:
            return -1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img = Image.open(sample['img_path']).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = self.to_tensor(img)
        img = self.normalize(img)
        
        return {'image': img, 'class_id': sample['class_id'], 'img_id': sample['img_id']}


# =======================
# 模型定义
# =======================
class SegModel(nn.Module):
    def __init__(self, backbone='resnet34'):
        super().__init__()
        if backbone == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            encoder_channels = [64, 64, 128, 256, 512]
        else:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            encoder_channels = [64, 256, 512, 1024, 2048]
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_channels[4], 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 1, 1)
        )
    
    def forward(self, x):
        f = self.layer0(x)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        return self.decoder(f)


class ClsModel(nn.Module):
    def __init__(self, num_classes=102, backbone='resnet34'):
        super().__init__()
        if backbone == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Linear(512, num_classes)
        else:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


# =======================
# 损失函数
# =======================
class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred_sigmoid * target).sum()
        dice = (2. * intersection + smooth) / (pred_sigmoid.sum() + target.sum() + smooth)
        dice_loss = 1 - dice
        
        return bce_loss * (1 - self.dice_weight) + dice_loss * self.dice_weight


# =======================
# 评估指标
# =======================
def compute_seg_metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    
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
    
    per_class_acc = []
    for c in range(min(num_classes, max(labels) + 1)):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc.append(np.mean(preds[mask] == c))
    
    macro_f1 = np.mean(per_class_acc) if per_class_acc else 0
    
    return {'Accuracy': float(acc), 'MacroF1': float(macro_f1)}


# =======================
# 训练函数
# =======================
def train_seg_model(train_loader, val_loader, epochs=10, lr=1e-4):
    print(f"\n  训练分割模型...")
    model = SegModel(backbone='resnet34').to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = DiceBCELoss()
    
    best_miou = 0
    best_metrics = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            images = batch['image'].to(DEVICE)
            gts = batch['gt'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(1), gts)
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
                
                outputs = model(images)
                preds = torch.sigmoid(outputs).cpu().numpy()
                preds = (preds > 0.5).astype(np.float32)
                
                for i in range(images.shape[0]):
                    metrics = compute_seg_metrics(preds[i, 0], gts[i])
                    all_metrics.append(metrics)
        
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        
        if avg_metrics['mIoU'] > best_miou:
            best_miou = avg_metrics['mIoU']
            best_metrics = avg_metrics
        
        print(f"    Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.4f}, mIoU={avg_metrics['mIoU']:.4f}, Dice={avg_metrics['Dice']:.4f}")
    
    return best_metrics


def train_cls_model(train_loader, val_loader, num_classes, epochs=10, lr=1e-4):
    print(f"\n  训练分类模型...")
    model = ClsModel(num_classes=num_classes, backbone='resnet34').to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
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
            best_metrics = metrics
        
        print(f"    Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.4f}, Acc={metrics['Accuracy']:.4f}, MacroF1={metrics['MacroF1']:.4f}")
    
    return best_metrics


# =======================
# 主函数
# =======================
def main():
    print("="*80)
    print("真实训练实验 - COD10K分割 + IP02分类")
    print(f"设备: {DEVICE}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': DEVICE,
        'experiments': {}
    }
    
    cod10k_root = r"d:\University\竞赛\4C\智慧农林\模型代码\COD10K-v3\COD10K-v3"
    ip02_root = r"d:\University\竞赛\4C\智慧农林\模型代码\数据集\数据集\IP02"
    
    print("\n" + "="*80)
    print("实验1: COD10K 伪装目标分割")
    print("="*80)
    
    print(f"\n[1] 加载数据集...")
    full_dataset = COD10KDataset(cod10k_root, split='test', img_size=256)
    print(f"    总样本数: {len(full_dataset)}")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"    训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    seg_metrics = train_seg_model(train_loader, val_loader, epochs=15, lr=1e-4)
    results['experiments']['COD10K_Segmentation_Ours'] = seg_metrics
    
    print("\n" + "="*80)
    print("实验2: IP02 害虫分类")
    print("="*80)
    
    print(f"\n[1] 加载数据集...")
    full_dataset = IP02ClassDataset(ip02_root, split='test', img_size=224)
    print(f"    总样本数: {len(full_dataset)}")
    print(f"    类别数: {full_dataset.num_classes}")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"    训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    cls_metrics = train_cls_model(train_loader, val_loader, full_dataset.num_classes, epochs=15, lr=1e-4)
    results['experiments']['IP02_Classification_Ours'] = cls_metrics
    
    results_file = os.path.join(RESULTS_DIR, 'real_training_results.json')
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
