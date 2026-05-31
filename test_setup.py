# -*- coding: utf-8 -*-
"""
测试脚本 - 验证模型和实验代码
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch

def test_model():
    """测试模型"""
    print("="*60)
    print("测试带分类能力的模型")
    print("="*60)
    
    from my_models.我的模型.model_with_classification import FewShotCamouflageSegWithClass, build_model_with_class
    
    print("\n[1] 构建模型...")
    model = build_model_with_class(num_classes=102)
    print("  模型构建成功!")
    
    print("\n[2] 测试前向传播...")
    K = 5
    B = 1
    H, W = 256, 256
    
    supp_imgs = torch.randn(K, 3, H, W)
    supp_masks = (torch.rand(K, 1, H, W) > 0.7).float()
    qry_imgs = torch.randn(B, 3, H, W)
    
    with torch.no_grad():
        output = model(supp_imgs, supp_masks, qry_imgs)
    
    print("  前向传播成功!")
    print(f"  输出键: {list(output.keys())}")
    print(f"  seg_logits shape: {output['seg_logits'].shape}")
    print(f"  class_logits shape: {output['class_logits'].shape}")
    print(f"  bbox shape: {output['bbox'].shape}")
    print(f"  confidence shape: {output['confidence'].shape}")
    
    print("\n[3] 测试检测输出...")
    detection_results = model.get_detection_output(output)
    print(f"  检测结果: {detection_results[0]}")
    
    print("\n模型测试通过!")
    return True


def test_experiments():
    """测试实验脚本"""
    print("\n" + "="*60)
    print("测试实验脚本")
    print("="*60)
    
    print("\n[1] 测试实验5.2导入...")
    try:
        from experiments.exp_5_2_detection_comparison import run_experiment_5_2
        print("  实验5.2导入成功!")
    except Exception as e:
        print(f"  实验5.2导入失败: {e}")
    
    print("\n[2] 测试实验5.3导入...")
    try:
        from experiments.exp_5_3_camouflage_analysis import run_experiment_5_3
        print("  实验5.3导入成功!")
    except Exception as e:
        print(f"  实验5.3导入失败: {e}")
    
    print("\n[3] 测试实验5.4导入...")
    try:
        from experiments.exp_5_4_fewshot_generalization import run_experiment_5_4
        print("  实验5.4导入成功!")
    except Exception as e:
        print(f"  实验5.4导入失败: {e}")
    
    return True


if __name__ == "__main__":
    print("\n开始测试...")
    
    try:
        test_model()
    except Exception as e:
        print(f"模型测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_experiments()
    except Exception as e:
        print(f"实验测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
