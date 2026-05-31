# -*- coding: utf-8 -*-
"""
农林虫害智检系统 - 推理脚本

支持：
1. 单张图像推理
2. 批量图像推理
3. 视频推理
4. 结果可视化

使用方法：
    python inference.py --image path/to/image.jpg --weights checkpoints/model_final.pth
    python inference.py --folder path/to/images --weights checkpoints/model_final.pth
    python inference.py --video path/to/video.mp4 --weights checkpoints/model_final.pth
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from my_models.我的模型.model_with_classification import FewShotCamouflageSegWithClass


class PestDetector:
    """虫害检测器"""
    
    def __init__(self, weights_path, device='cuda', num_classes=102):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        print(f"加载模型: {weights_path}")
        print(f"使用设备: {self.device}")
        
        self.model = FewShotCamouflageSegWithClass(
            backbone='resnet50',
            num_classes=num_classes,
            use_adapter=True
        ).to(self.device)
        
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            print("模型权重加载成功")
        else:
            print(f"警告: 权重文件不存在: {weights_path}")
            print("使用随机初始化权重")
        
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = self._load_class_names()
        
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
    
    def _load_class_names(self):
        class_names = [f"class_{i}" for i in range(self.num_classes)]
        pest_names = [
            "稻飞虱", "二化螟", "三化螟", "稻纵卷叶螟", "稻苞虫",
            "褐飞虱", "白背飞虱", "灰飞虱", "稻蓟马", "稻蝗",
            "粘虫", "玉米螟", "棉铃虫", "棉蚜", "红蜘蛛",
            "蚜虫", "粉虱", "蓟马", "叶蝉", "飞虱"
        ]
        for i, name in enumerate(pest_names):
            if i < self.num_classes:
                class_names[i] = name
        return class_names
    
    def preprocess(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        original_size = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return image_tensor, original_size
    
    def infer_single(self, image, support_images=None, support_masks=None):
        image_tensor, original_size = self.preprocess(image)
        
        if support_images is None:
            support_images = image_tensor.repeat(5, 1, 1, 1)
            support_masks = torch.zeros(5, 1, 640, 640, device=self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor, support_images, support_masks)
        
        seg_mask = torch.sigmoid(outputs['seg_logits']).squeeze().cpu().numpy()
        seg_mask = (seg_mask > 0.5).astype(np.uint8) * 255
        
        seg_mask = cv2.resize(seg_mask, original_size, interpolation=cv2.INTER_LINEAR)
        
        cls_probs = F.softmax(outputs['cls_logits'], dim=1).squeeze().cpu().numpy()
        pred_class = np.argmax(cls_probs)
        confidence = cls_probs[pred_class]
        
        bbox = outputs['bbox'].squeeze().cpu().numpy()
        
        return {
            'mask': seg_mask,
            'class_id': int(pred_class),
            'class_name': self.class_names[pred_class],
            'confidence': float(confidence),
            'bbox': bbox,
            'all_probs': cls_probs
        }
    
    def draw_results(self, image, results, output_path=None):
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        h, w = image.shape[:2]
        
        mask = results['mask']
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        mask_colored = cv2.addWeighted(image, 0.6, mask_colored, 0.4, 0)
        
        bbox = results['bbox']
        x_center, y_center, box_w, box_h = bbox
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)
        
        color = self.colors[results['class_id'] % len(self.colors)]
        cv2.rectangle(mask_colored, (x1, y1), (x2, y2), color, 2)
        
        label = f"{results['class_name']}: {results['confidence']:.2f}"
        cv2.putText(mask_colored, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        info_text = f"Class: {results['class_name']} | Conf: {results['confidence']:.3f}"
        cv2.putText(mask_colored, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, mask_colored)
            print(f"结果已保存: {output_path}")
        
        return mask_colored
    
    def infer_folder(self, folder_path, output_dir, save_json=True):
        folder_path = Path(folder_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f'*{ext}'))
            image_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        print(f"找到 {len(image_files)} 张图像")
        
        all_results = []
        
        for img_path in tqdm(image_files, desc="推理中"):
            results = self.infer_single(str(img_path))
            
            output_path = output_dir / f"{img_path.stem}_result.jpg"
            self.draw_results(str(img_path), results, str(output_path))
            
            all_results.append({
                'image': str(img_path),
                'class_id': results['class_id'],
                'class_name': results['class_name'],
                'confidence': results['confidence'],
                'bbox': results['bbox'].tolist()
            })
        
        if save_json:
            json_path = output_dir / 'results.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存: {json_path}")
        
        return all_results
    
    def infer_video(self, video_path, output_path, fps=None):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps is None:
            fps = original_fps
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"处理视频: {video_path}")
        print(f"分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.infer_single(frame)
            result_frame = self.draw_results(frame, results)
            
            out.write(result_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                remaining = (total_frames - frame_count) / (frame_count / elapsed)
                print(f"进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%), "
                      f"预计剩余: {remaining:.1f}s")
        
        cap.release()
        out.release()
        
        total_time = time.time() - start_time
        print(f"\n视频处理完成!")
        print(f"总帧数: {frame_count}")
        print(f"总时间: {total_time:.1f}s")
        print(f"平均FPS: {frame_count/total_time:.1f}")
        print(f"输出文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='农林虫害智检系统 - 推理脚本')
    
    parser.add_argument('--weights', type=str, 
                       default='checkpoints/demo1_improved_model.pth',
                       help='模型权重路径')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='推理设备')
    
    parser.add_argument('--image', type=str, default='',
                       help='单张图像路径')
    parser.add_argument('--folder', type=str, default='',
                       help='图像文件夹路径')
    parser.add_argument('--video', type=str, default='',
                       help='视频文件路径')
    
    parser.add_argument('--output', type=str, default='output',
                       help='输出目录')
    parser.add_argument('--num_classes', type=int, default=102,
                       help='类别数量')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"农林虫害智检系统 - 推理脚本")
    print(f"{'='*60}\n")
    
    detector = PestDetector(args.weights, args.device, args.num_classes)
    
    if args.image:
        print(f"\n处理单张图像: {args.image}")
        results = detector.infer_single(args.image)
        
        print(f"\n检测结果:")
        print(f"  类别: {results['class_name']} (ID: {results['class_id']})")
        print(f"  置信度: {results['confidence']:.3f}")
        print(f"  边界框: {results['bbox']}")
        
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{Path(args.image).stem}_result.jpg"
        detector.draw_results(args.image, results, str(output_file))
    
    elif args.folder:
        print(f"\n处理图像文件夹: {args.folder}")
        detector.infer_folder(args.folder, args.output)
    
    elif args.video:
        print(f"\n处理视频: {args.video}")
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{Path(args.video).stem}_result.mp4"
        detector.infer_video(args.video, str(output_file))
    
    else:
        print("请指定输入: --image, --folder 或 --video")
        print("\n示例:")
        print("  python inference.py --image test.jpg --weights model.pth")
        print("  python inference.py --folder ./images --weights model.pth")
        print("  python inference.py --video test.mp4 --weights model.pth")


if __name__ == '__main__':
    main()
