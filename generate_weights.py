# -*- coding: utf-8 -*-
"""
权重生成脚本 - 用于快速生成测试权重

当训练权重丢失时，可使用此脚本生成随机初始化的权重文件，
以便测试推理脚本是否正常工作。

注意：生成的权重是随机初始化的，不包含任何学习到的知识，
仅用于测试代码流程，不能用于实际推理。

使用方法：
    python generate_weights.py --output checkpoints/model_final.pth
    python generate_weights.py --all  # 生成所有需要的权重
"""
import os
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from my_models.我的模型.model_with_classification import FewShotCamouflageSegWithClass


def generate_single_weight(output_path, backbone='resnet50', num_classes=102):
    """生成单个权重文件"""
    print(f"生成权重: {output_path}")
    
    model = FewShotCamouflageSegWithClass(
        backbone=backbone,
        num_classes=num_classes,
        use_adapter=True
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_path)
    print(f"权重已保存: {output_path}")
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"文件大小: {file_size:.2f} MB")
    
    return output_path


def generate_all_weights(base_dir='checkpoints'):
    """生成所有需要的权重文件"""
    base_dir = Path(base_dir)
    
    configs = [
        ('fewshot_1shot_COD10K/model_final.pth', 'resnet50', 102),
        ('fewshot_3shot_COD10K/model_final.pth', 'resnet50', 102),
        ('fewshot_5shot_COD10K/model_final.pth', 'resnet50', 102),
        ('fewshot_20shot_COD10K/model_final.pth', 'resnet50', 102),
        ('full_IP102/model_final.pth', 'resnet50', 102),
        ('ablation_1/model_final.pth', 'resnet50', 102),
        ('ablation_2/model_final.pth', 'resnet50', 102),
        ('ablation_3/model_final.pth', 'resnet50', 102),
        ('ablation_4/model_final.pth', 'resnet50', 102),
        ('ablation_5/model_final.pth', 'resnet50', 102),
        ('ablation_6/model_final.pth', 'resnet50', 102),
    ]
    
    print(f"\n{'='*60}")
    print(f"生成所有权重文件")
    print(f"{'='*60}\n")
    
    generated = []
    for rel_path, backbone, num_classes in configs:
        output_path = base_dir / rel_path
        generate_single_weight(output_path, backbone, num_classes)
        generated.append(str(output_path))
    
    print(f"\n{'='*60}")
    print(f"生成完成！共 {len(generated)} 个权重文件")
    print(f"{'='*60}\n")
    
    return generated


def verify_weight(weight_path):
    """验证权重文件"""
    print(f"\n验证权重: {weight_path}")
    
    if not os.path.exists(weight_path):
        print(f"错误: 文件不存在")
        return False
    
    try:
        state_dict = torch.load(weight_path, map_location='cpu')
        
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"参数数量: {total_params:,}")
        
        print("前5个键:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            shape = state_dict[key].shape
            print(f"  {key}: {shape}")
        
        print("验证通过!")
        return True
        
    except Exception as e:
        print(f"验证失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='权重生成脚本')
    
    parser.add_argument('--output', type=str, default='',
                       help='输出权重路径')
    parser.add_argument('--all', action='store_true',
                       help='生成所有需要的权重')
    parser.add_argument('--verify', type=str, default='',
                       help='验证权重文件')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       help='骨干网络')
    parser.add_argument('--num_classes', type=int, default=102,
                       help='类别数量')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_weight(args.verify)
    elif args.all:
        generate_all_weights()
    elif args.output:
        generate_single_weight(args.output, args.backbone, args.num_classes)
    else:
        print("请指定操作:")
        print("  --output PATH  生成单个权重")
        print("  --all          生成所有权重")
        print("  --verify PATH  验证权重文件")
        print("\n示例:")
        print("  python generate_weights.py --output checkpoints/model_final.pth")
        print("  python generate_weights.py --all")
        print("  python generate_weights.py --verify checkpoints/model_final.pth")


if __name__ == '__main__':
    main()
