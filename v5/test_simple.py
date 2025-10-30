#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for discPC complex datasets
"""

import torch
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def test_basic_functionality():
    """Test basic functionality without complex features"""
    print("Testing basic functionality...")
    
    # Test data loading
    print("1. Loading Wine dataset...")
    wine = load_wine()
    X = wine.data
    y = wine.target.reshape(-1, 1)
    print(f"   Data shape: {X.shape}, Labels shape: {y.shape}")
    
    # Test preprocessing
    print("2. Testing preprocessing...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   Scaled data shape: {X_scaled.shape}")
    
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)
    print(f"   Encoded labels shape: {y_encoded.shape}")
    
    # Test train/test split
    print("3. Testing train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Test PyTorch conversion
    print("4. Testing PyTorch conversion...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device).T
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device).T
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).T
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device).T
    
    print(f"   Tensor shapes: X_train {X_train_tensor.shape}, y_train {y_train_tensor.shape}")
    
    print("✓ All basic tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
        print("\nBasic functionality test completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
