# -*- coding: utf-8 -*-
"""
快速训练脚本 - 用于IDE环境

使用更小的模型和更少的epoch，快速获得真实结果
"""
import os
import sys
import json
import time
import random
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
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


class COD10KDataset(Dataset):
    def __init__(self, root_dir, img_size=128, max_samples=100):
        self.img_size = img_size
        self.img_dir = os.path.join(root_dir, 'Test', 'Image')
        self.gt_dir = os.path.join(root_dir, 'Test', 'GT_Object')
        
        self.samples = []
        for fname in os.listdir(self.img_dir)[:max_samples]:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.img_dir, fname)
                gt_name = fname.replace('.jpg', '.png').replace('.jpeg', '.png')
                gt_path = os.path.join(self.gt_dir, gt_name)
                
                if os.path.exists(gt_path):
                    self.samples.append({'img_path': img_path, 'gt_path': gt_path})
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img_path']).convert("RGB").resize((self.img_size, self.img_size))
        img = self.normalize(self.to_tensor(img))
        
        gt = Image.open(sample['gt_path']).convert("L").resize((self.img_size, self.img_size))
        gt = torch.from_numpy((np.array(gt) > 127).astype(np.float32))
        
        return {'image': img, 'gt': gt}


class IP02Dataset(Dataset):
    def __init__(self, root_dir, img_size=128, max_samples=200):
        self.img_size = img_size
        det_dir = os.path.join(root_dir, "Detection-20260318T092509Z-1-001", "Detection", "VOC2007")
        self.img_dir = os.path.join(det_dir, "JPEGImages", "JPEGImages")
        
        self.samples = []
        for fname in os.listdir(self.img_dir)[:max_samples]:
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
        
        self.num_classes = max(s['class_id'] for s in self.samples) + 1 if self.samples else 50
        print(f"    加载了 {len(self.samples)} 个样本, {self.num_classes} 个类别")
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img_path']).convert("RGB").resize((self.img_size, self.img_size))
        img = self.normalize(self.to_tensor(img))
        return {'image': img, 'class_id': sample['class_id']}


class TinySegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, 1)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


class TinyClsModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        return self.fc(self.features(x).view(x.size(0), -1))


def compute_seg_metrics(pred, gt):
    pred, gt = pred.flatten(), gt.flatten()
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    return {'mIoU': float(iou), 'Dice': float(dice), 'Precision': float(prec), 'Recall': float(rec), 'F1': float(2*prec*rec/(prec+rec+1e-6))}


def main():
    print("="*60)
    print("快速训练实验")
    print(f"设备: {DEVICE}")
    print("="*60)
    
    results = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'device': DEVICE, 'experiments': {}}
    
    cod10k_root = r"d:\University\竞赛\4C\智慧农林\模型代码\COD10K-v3\COD10K-v3"
    ip02_root = r"d:\University\竞赛\4C\智慧农林\模型代码\数据集\数据集\IP02"
    
    print("\n[1] COD10K分割训练...")
    dataset = COD10KDataset(cod10k_root, img_size=128, max_samples=80)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)
    
    model = TinySegModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(5):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch['image'].to(DEVICE)).squeeze(1), batch['gt'].to(DEVICE))
            loss.backward()
            optimizer.step()
        
        model.eval()
        metrics_list = []
        with torch.no_grad():
            for batch in val_loader:
                preds = (torch.sigmoid(model(batch['image'].to(DEVICE))) > 0.5).cpu().numpy()
                for i in range(len(preds)):
                    metrics_list.append(compute_seg_metrics(preds[i, 0], batch['gt'][i].numpy()))
        
        avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        print(f"  Epoch {epoch+1}: mIoU={avg['mIoU']:.4f}, Dice={avg['Dice']:.4f}")
    
    results['experiments']['COD10K_Segmentation'] = avg
    
    print("\n[2] IP02分类训练...")
    dataset = IP02Dataset(ip02_root, img_size=128, max_samples=200)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)
    
    model = TinyClsModel(dataset.num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch['image'].to(DEVICE)), batch['class_id'].to(DEVICE))
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch['image'].to(DEVICE)).argmax(1).cpu()
                correct += (preds == batch['class_id']).sum().item()
                total += len(preds)
        
        acc = correct / total
        print(f"  Epoch {epoch+1}: Acc={acc:.4f}")
    
    results['experiments']['IP02_Classification'] = {'Accuracy': acc}
    
    results_file = os.path.join(RESULTS_DIR, 'quick_training_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"结果保存到: {results_file}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
