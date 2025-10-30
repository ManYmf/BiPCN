#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的导入问题
"""

def test_imports():
    """测试所有导入是否正常"""
    try:
        print("测试导入...")
        
        import torch
        print("✓ torch 导入成功")
        
        import numpy as np
        print("✓ numpy 导入成功")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib 导入成功")
        
        from sklearn.datasets import load_wine, load_breast_cancer, load_digits
        print("✓ sklearn.datasets 导入成功")
        
        from sklearn.model_selection import train_test_split
        print("✓ sklearn.model_selection 导入成功")
        
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        print("✓ sklearn.preprocessing 导入成功")
        
        from sklearn.metrics import classification_report, confusion_matrix
        print("✓ sklearn.metrics 导入成功")
        
        import seaborn as sns
        print("✓ seaborn 导入成功")
        
        import time
        print("✓ time 导入成功")
        
        print("\n所有导入测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_data_loading():
    """测试数据加载功能"""
    try:
        print("\n测试数据加载...")
        
        from sklearn.datasets import load_wine, load_breast_cancer, load_digits
        
        # 测试Wine数据集
        wine = load_wine()
        print(f"✓ Wine数据集: {wine.data.shape}, {wine.target.shape}")
        
        # 测试Breast Cancer数据集
        cancer = load_breast_cancer()
        print(f"✓ Breast Cancer数据集: {cancer.data.shape}, {cancer.target.shape}")
        
        # 测试Digits数据集
        digits = load_digits()
        print(f"✓ Digits数据集: {digits.data.shape}, {digits.target.shape}")
        
        print("\n数据加载测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False

def test_pytorch():
    """测试PyTorch功能"""
    try:
        print("\n测试PyTorch...")
        
        import torch
        import numpy as np
        
        # 测试设备检测
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ 使用设备: {device}")
        
        # 测试张量创建
        x = torch.randn(10, 5)
        print(f"✓ 张量创建: {x.shape}")
        
        # 测试激活函数
        sigmoid_out = torch.sigmoid(x)
        print(f"✓ Sigmoid激活: {sigmoid_out.shape}")
        
        print("\nPyTorch测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ PyTorch测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("discPC复杂数据集测试 - 环境检查")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_data_loading()
    success &= test_pytorch()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ 所有测试通过！环境配置正确。")
        print("现在可以运行完整的复杂数据集测试了。")
    else:
        print("✗ 部分测试失败，请检查环境配置。")
    print("=" * 50)
