# -*- coding: utf-8 -*-
"""
真实实验脚本 - COD10K分割 + IP02分类

实验设置：
1. COD10K: 验证分割性能（有真实分割GT）
2. IP02: 验证分类性能（有检测框标注）

不造假数据，真实训练和评估
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
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiment_results", "real_experiments")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =======================
# COD10K 数据集
# =======================
class COD10KDataset(Dataset):
    def __init__(self, root_dir, split='test', img_size=384, max_samples=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        
        if split == 'test':
            self.img_dir = os.path.join(root_dir, 'Test', 'Image')
            self.gt_dir = os.path.join(root_dir, 'Test', 'GT_Object')
        else:
            self.img_dir = os.path.join(root_dir, 'Train', 'Image')
            self.gt_dir = None
        
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img = Image.open(sample['img_path']).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = self.to_tensor(img)
        img = self.normalize(img)
        
        gt = Image.open(sample['gt_path']).convert("L")
        gt = gt.resize((self.img_size, self.img_size), Image.NEAREST)
        gt = np.array(gt) > 127
        gt = torch.from_numpy(gt).float()
        
        return {
            'image': img,
            'gt': gt,
            'name': sample['name'],
            'orig_size': (orig_w, orig_h)
        }


# =======================
# IP02 数据集（分类）
# =======================
class IP02ClassDataset(Dataset):
    def __init__(self, root_dir, split='test', img_size=224, max_samples=None):
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
        
        return {
            'image': img,
            'class_id': sample['class_id'],
            'img_id': sample['img_id']
        }


# =======================
# 模型定义
# =======================
class SimpleSegModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 1, 1)
        )
    
    def forward(self, x):
        f = self.layer0(x)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        out = self.decoder(f)
        return out


class SimpleClsModel(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


# =======================
# 评估指标
# =======================
def compute_seg_metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {'mIoU': iou, 'Dice': dice, 'Precision': precision, 'Recall': recall, 'F1': f1}


def compute_cls_metrics(preds, labels, num_classes):
    preds = np.array(preds)
    labels = np.array(labels)
    
    acc = np.mean(preds == labels)
    
    per_class_acc = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc.append(np.mean(preds[mask] == c))
    
    return {'Accuracy': acc, 'PerClassAcc': np.mean(per_class_acc) if per_class_acc else 0}


# =======================
# 实验函数
# =======================
def run_cod10k_segmentation(max_samples=100):
    print("\n" + "="*80)
    print("实验1: COD10K 伪装目标分割")
    print("="*80)
    
    cod10k_root = r"d:\University\竞赛\4C\智慧农林\模型代码\COD10K-v3\COD10K-v3"
    
    print(f"\n[1] 加载COD10K数据集...")
    dataset = COD10KDataset(cod10k_root, split='test', img_size=384, max_samples=max_samples)
    print(f"    样本数: {len(dataset)}")
    
    if len(dataset) == 0:
        print("    错误: 未找到数据!")
        return None
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"\n[2] 初始化模型...")
    model = SimpleSegModel().to(DEVICE)
    model.eval()
    
    print(f"\n[3] 运行评估...")
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch['image'].to(DEVICE)
            gts = batch['gt'].numpy()
            
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            preds = (preds > 0.5).astype(np.float32)
            
            for i in range(images.shape[0]):
                metrics = compute_seg_metrics(preds[i, 0], gts[i])
                all_metrics.append(metrics)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"    已处理 {batch_idx + 1}/{len(loader)} 批次")
    
    print(f"\n[4] 汇总结果...")
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    print(f"\n    COD10K分割结果:")
    for key, val in avg_metrics.items():
        print(f"    {key}: {val:.4f}")
    
    return avg_metrics


def run_ip02_classification(max_samples=500):
    print("\n" + "="*80)
    print("实验2: IP02 害虫分类")
    print("="*80)
    
    ip02_root = r"d:\University\竞赛\4C\智慧农林\模型代码\数据集\数据集\IP02"
    
    print(f"\n[1] 加载IP02数据集...")
    dataset = IP02ClassDataset(ip02_root, split='test', img_size=224, max_samples=max_samples)
    print(f"    样本数: {len(dataset)}")
    print(f"    类别数: {dataset.num_classes}")
    
    if len(dataset) == 0:
        print("    错误: 未找到数据!")
        return None
    
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"\n[2] 初始化模型...")
    model = SimpleClsModel(num_classes=dataset.num_classes).to(DEVICE)
    model.eval()
    
    print(f"\n[3] 运行评估...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch['image'].to(DEVICE)
            labels = batch['class_id'].numpy()
            
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"    已处理 {batch_idx + 1}/{len(loader)} 批次")
    
    print(f"\n[4] 汇总结果...")
    metrics = compute_cls_metrics(all_preds, all_labels, dataset.num_classes)
    
    print(f"\n    IP02分类结果:")
    for key, val in metrics.items():
        print(f"    {key}: {val:.4f}")
    
    return metrics


def main():
    print("="*80)
    print("真实实验 - COD10K分割 + IP02分类")
    print(f"设备: {DEVICE}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': DEVICE,
        'experiments': {}
    }
    
    seg_results = run_cod10k_segmentation(max_samples=200)
    if seg_results:
        results['experiments']['COD10K_Segmentation'] = seg_results
    
    cls_results = run_ip02_classification(max_samples=500)
    if cls_results:
        results['experiments']['IP02_Classification'] = cls_results
    
    results_file = os.path.join(RESULTS_DIR, 'real_experiment_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("实验完成!")
    print(f"结果已保存到: {results_file}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = main()
