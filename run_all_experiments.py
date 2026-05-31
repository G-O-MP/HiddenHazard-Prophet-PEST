# -*- coding: utf-8 -*-
"""
实验运行脚本 - 统一入口

运行所有实验：
- 实验5.2：模型检测性能对比分析
- 实验5.3：伪装目标检测效果分析
- 实验5.4：小样本泛化能力验证

使用方法：
    python run_all_experiments.py --exp 5.2
    python run_all_experiments.py --exp 5.3
    python run_all_experiments.py --exp 5.4
    python run_all_experiments.py --exp all
"""
import os
import sys
import argparse
import json
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiment_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp_5_2():
    """运行实验5.2：模型检测性能对比分析"""
    print("\n" + "="*80)
    print("开始运行实验5.2：模型检测性能对比分析")
    print("="*80)
    
    try:
        from experiments.exp_5_2_detection_comparison import run_experiment_5_2
        results = run_experiment_5_2()
        return results
    except Exception as e:
        print(f"实验5.2运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_exp_5_3():
    """运行实验5.3：伪装目标检测效果分析"""
    print("\n" + "="*80)
    print("开始运行实验5.3：伪装目标检测效果分析")
    print("="*80)
    
    try:
        from experiments.exp_5_3_camouflage_analysis import run_experiment_5_3
        results = run_experiment_5_3()
        return results
    except Exception as e:
        print(f"实验5.3运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_exp_5_4():
    """运行实验5.4：小样本泛化能力验证"""
    print("\n" + "="*80)
    print("开始运行实验5.4：小样本泛化能力验证")
    print("="*80)
    
    try:
        from experiments.exp_5_4_fewshot_generalization import run_experiment_5_4
        results = run_experiment_5_4()
        return results
    except Exception as e:
        print(f"实验5.4运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_summary_report(all_results):
    """生成汇总报告"""
    print("\n" + "="*80)
    print("实验汇总报告")
    print("="*80)
    
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': {}
    }
    
    if '5.2' in all_results and all_results['5.2']:
        report['experiments']['5.2'] = {
            'name': '模型检测性能对比分析',
            'status': 'completed',
            'summary': {}
        }
        
        if 'methods' in all_results['5.2']:
            for method, metrics in all_results['5.2']['methods'].items():
                report['experiments']['5.2']['summary'][method] = {
                    'mIoU': metrics.get('mIoU', 0),
                    'Dice': metrics.get('Dice', 0),
                    'F1': metrics.get('F1', metrics.get('Dice', 0))
                }
    
    if '5.3' in all_results and all_results['5.3']:
        report['experiments']['5.3'] = {
            'name': '伪装目标检测效果分析',
            'status': 'completed',
            'summary': {}
        }
        
        if 'scenarios' in all_results['5.3']:
            for scenario, methods in all_results['5.3']['scenarios'].items():
                report['experiments']['5.3']['summary'][scenario] = {
                    'Ours_F1': methods.get('Ours', {}).get('F1', 0),
                    'Ours_Recall': methods.get('Ours', {}).get('Recall', 0)
                }
    
    if '5.4' in all_results and all_results['5.4']:
        report['experiments']['5.4'] = {
            'name': '小样本泛化能力验证',
            'status': 'completed',
            'summary': {}
        }
        
        if 'k_shot_results' in all_results['5.4']:
            for k_shot, methods in all_results['5.4']['k_shot_results'].items():
                report['experiments']['5.4']['summary'][k_shot] = {
                    'Ours_F1': methods.get('Ours', {}).get('F1', 0),
                    'Ours_mIoU': methods.get('Ours', {}).get('mIoU', 0)
                }
    
    report_file = os.path.join(RESULTS_DIR, 'all_experiments_summary.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n汇总报告已保存到: {report_file}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='运行实验')
    parser.add_argument('--exp', type=str, default='all', 
                        choices=['5.2', '5.3', '5.4', 'all'],
                        help='要运行的实验编号')
    args = parser.parse_args()
    
    print("="*80)
    print("农林虫害智检系统 - 实验运行脚本")
    print("="*80)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"实验选择: {args.exp}")
    
    all_results = {}
    
    if args.exp in ['5.2', 'all']:
        start_time = time.time()
        all_results['5.2'] = run_exp_5_2()
        print(f"实验5.2耗时: {time.time() - start_time:.2f}秒")
    
    if args.exp in ['5.3', 'all']:
        start_time = time.time()
        all_results['5.3'] = run_exp_5_3()
        print(f"实验5.3耗时: {time.time() - start_time:.2f}秒")
    
    if args.exp in ['5.4', 'all']:
        start_time = time.time()
        all_results['5.4'] = run_exp_5_4()
        print(f"实验5.4耗时: {time.time() - start_time:.2f}秒")
    
    if args.exp == 'all':
        generate_summary_report(all_results)
    
    print("\n" + "="*80)
    print("实验运行完成!")
    print("="*80)


if __name__ == "__main__":
    main()
