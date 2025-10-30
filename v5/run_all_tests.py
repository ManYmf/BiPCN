#!/usr/bin/env python3
"""
discPC综合测试脚本
包含复杂数据集测试、性能基准对比、超参数调优等功能
"""

import sys
import os
import argparse
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='discPC综合测试脚本')
    parser.add_argument('--test', choices=['complex', 'benchmark', 'tuning', 'all'], 
                       default='all', help='选择要运行的测试类型')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['wine', 'breast_cancer', 'digits'],
                       default=['wine', 'breast_cancer', 'digits'],
                       help='选择要测试的数据集')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（减少参数组合数量）')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'],
                       default='auto', help='选择计算设备')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    print(f"测试模式: {args.test}")
    print(f"数据集: {args.datasets}")
    print(f"快速模式: {args.quick}")
    print("=" * 60)
    
    start_time = time.time()
    
    if args.test in ['complex', 'all']:
        print("\n1. 运行复杂数据集测试...")
        try:
            from v5_complex_datasets import test_complex_datasets
            complex_results = test_complex_datasets()
            print("✓ 复杂数据集测试完成")
        except Exception as e:
            print(f"✗ 复杂数据集测试失败: {e}")
    
    if args.test in ['benchmark', 'all']:
        print("\n2. 运行性能基准对比...")
        try:
            from v5_benchmark import run_benchmark
            benchmark_results = run_benchmark()
            print("✓ 性能基准对比完成")
        except Exception as e:
            print(f"✗ 性能基准对比失败: {e}")
    
    if args.test in ['tuning', 'all']:
        print("\n3. 运行超参数调优...")
        try:
            from v5_hyperparameter_tuning import run_hyperparameter_tuning
            tuning_results = run_hyperparameter_tuning()
            print("✓ 超参数调优完成")
        except Exception as e:
            print(f"✗ 超参数调优失败: {e}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"所有测试完成！总用时: {total_time:.2f}秒")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
