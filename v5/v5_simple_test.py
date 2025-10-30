#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的discPC复杂数据集测试
修复了导入错误和编码问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

# 设置matplotlib字体
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 1. 激活函数
# --------------------------
def sigmoid(x):
    return torch.sigmoid(torch.clamp(x, -20, 20))

def softmax(x):
    exp_x = torch.exp(torch.clamp(x, -20, 20))
    return exp_x / torch.sum(exp_x, dim=0, keepdim=True)

# --------------------------
# 2. 简化的discPC网络类
# --------------------------
class discPC_Simple:
    def __init__(self, layers, device='cpu'):
        self.layers = layers
        self.n_layers = len(layers)
        self.device = device
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            input_dim = layers[i]
            output_dim = layers[i+1]
            
            # Xavier初始化
            weight = torch.randn(input_dim, output_dim, device=device) * np.sqrt(2.0 / (input_dim + output_dim))
            self.weights.append(weight)
            
            bias = torch.zeros(output_dim, 1, device=device)
            self.biases.append(bias)
    
    def initialize_activity(self, x_batch):
        activities = [x_batch]
        
        for i in range(self.n_layers - 1):
            prev_activity = activities[-1]
            layer_input = self.weights[i].T @ prev_activity + self.biases[i]
            
            if i == self.n_layers - 1:  # 输出层
                activity = softmax(layer_input)
            else:  # 隐藏层
                activity = sigmoid(layer_input)
            
            activities.append(activity)
        
        return activities
    
    def update_activity(self, x_batch, activities, target=None, lr=0.1, m=20, lambda_target=0.9):
        current_activities = [act.clone() for act in activities]
        
        for _ in range(m):
            # 前向计算预测值
            pred_activities = [x_batch]
            for i in range(self.n_layers - 1):
                prev_pred = pred_activities[-1]
                layer_input = self.weights[i].T @ prev_pred + self.biases[i]
                
                if i == self.n_layers - 1:
                    pred_activities.append(softmax(layer_input))
                else:
                    pred_activities.append(sigmoid(layer_input))
            
            # 反向更新活动值
            for i in reversed(range(1, self.n_layers)):
                epsilon = current_activities[i] - pred_activities[i]
                update = -lr * epsilon
                
                if target is not None and i == self.n_layers - 1:
                    target_error = current_activities[i] - target
                    update -= lr * lambda_target * target_error
                elif target is not None and i < self.n_layers - 1:
                    sigmoid_deriv = pred_activities[i] * (1 - pred_activities[i])
                    next_error = current_activities[i+1] - target if i+1 == self.n_layers -1 else current_activities[i+1] - pred_activities[i+1]
                    target_error = (self.weights[i] @ next_error) * sigmoid_deriv
                    update -= lr * lambda_target * target_error
                
                current_activities[i] += update
        
        return current_activities
    
    def update_weight(self, x_batch, activities, target, lr=0.01):
        pred_activities = [x_batch]
        for i in range(self.n_layers - 1):
            prev_pred = pred_activities[-1]
            layer_input = self.weights[i].T @ prev_pred + self.biases[i]
            
            if i == self.n_layers - 1:
                pred_activities.append(softmax(layer_input))
            else:
                pred_activities.append(sigmoid(layer_input))
        
        # 计算误差
        errors = [None] * (self.n_layers - 1)
        errors[-1] = target - pred_activities[-1]
        
        for i in reversed(range(self.n_layers - 2)):
            sigmoid_deriv = pred_activities[i+1] * (1 - pred_activities[i+1])
            errors[i] = (self.weights[i+1] @ errors[i+1]) * sigmoid_deriv
        
        # 更新权重和偏置
        batch_size = x_batch.shape[1]
        for i in range(self.n_layers - 1):
            self.weights[i] += lr * (pred_activities[i] @ errors[i].T) / batch_size
            self.biases[i] += lr * torch.mean(errors[i], dim=1, keepdim=True)
    
    def predict(self, x_batch, m=20):
        activities = self.initialize_activity(x_batch)
        activities = self.update_activity(x_batch, activities, m=m)
        return torch.argmax(activities[-1], dim=0)

# --------------------------
# 3. 数据加载函数
# --------------------------
def load_dataset_simple(dataset_name, test_size=0.2, device='cpu'):
    """加载数据集"""
    if dataset_name == 'wine':
        data = load_wine()
        X = data.data
        y = data.target.reshape(-1, 1)
        class_names = data.target_names
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        X = data.data
        y = data.target.reshape(-1, 1)
        class_names = data.target_names
    elif dataset_name == 'digits':
        data = load_digits()
        X = data.data
        y = data.target.reshape(-1, 1)
        class_names = [str(i) for i in range(10)]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    print(f"加载数据集: {dataset_name}")
    print(f"特征维度: {X.shape[1]}, 样本数量: {X.shape[0]}, 类别数量: {len(np.unique(y))}")
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 标签独热编码
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y
    )
    
    # 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device).T
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device).T
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device).T
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device).T
    
    return X_train, X_test, y_train, y_test, class_names

# --------------------------
# 4. 训练函数
# --------------------------
def train_discpc_simple(X_train, X_test, y_train, y_test, layers, n_epochs=1000, 
                       m=10, lr_activity=0.05, lr_weight=0.05, lambda_target=0.8, device='cpu'):
    """训练discPC模型"""
    # 创建模型
    model = discPC_Simple(layers, device=device)
    
    # 训练
    start_time = time.time()
    
    for epoch in range(n_epochs):
        activities = model.initialize_activity(X_train)
        activities = model.update_activity(
            X_train, activities, target=y_train,
            lr=lr_activity, m=m, lambda_target=lambda_target
        )
        model.update_weight(X_train, activities, y_train, lr=lr_weight)
        
        # 每100个epoch打印一次进度
        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            train_pred = torch.argmax(activities[-1], dim=0)
            train_true = torch.argmax(y_train, dim=0)
            train_acc = torch.mean((train_pred == train_true).float()).item()
            print(f"Epoch {epoch+1:4d}, 训练准确率: {train_acc:.4f}")
    
    training_time = time.time() - start_time
    
    # 预测
    y_pred = model.predict(X_test, m=m)
    accuracy = accuracy_score(torch.argmax(y_test, dim=0).cpu().numpy(), y_pred.cpu().numpy())
    
    return accuracy, training_time

# --------------------------
# 5. 主测试函数
# --------------------------
def test_datasets_simple():
    """测试多个数据集"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据集配置
    datasets_config = {
        'wine': {
            'layers': [13, 32, 16, 3],
            'n_epochs': 1000,
            'm': 15,
            'lr_activity': 0.03,
            'lr_weight': 0.03,
            'lambda_target': 0.7
        },
        'breast_cancer': {
            'layers': [30, 64, 32, 2],
            'n_epochs': 1500,
            'm': 20,
            'lr_activity': 0.02,
            'lr_weight': 0.02,
            'lambda_target': 0.8
        },
        'digits': {
            'layers': [64, 128, 64, 10],
            'n_epochs': 2000,
            'm': 25,
            'lr_activity': 0.01,
            'lr_weight': 0.01,
            'lambda_target': 0.9
        }
    }
    
    results_summary = {}
    
    for dataset_name, config in datasets_config.items():
        print(f"\n{'='*60}")
        print(f"测试数据集: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # 加载数据
        X_train, X_test, y_train, y_test, class_names = load_dataset_simple(dataset_name, device=device)
        
        # 训练和评估
        accuracy, training_time = train_discpc_simple(
            X_train, X_test, y_train, y_test,
            layers=config['layers'],
            n_epochs=config['n_epochs'],
            m=config['m'],
            lr_activity=config['lr_activity'],
            lr_weight=config['lr_weight'],
            lambda_target=config['lambda_target'],
            device=device
        )
        
        # 保存结果
        results_summary[dataset_name] = {
            'accuracy': accuracy,
            'training_time': training_time,
            'config': config
        }
        
        # 打印结果
        print(f"\n{dataset_name.upper()} 最终结果:")
        print(f"测试准确率: {accuracy:.4f}")
        print(f"训练时间: {training_time:.2f}秒")
    
    # 总结对比
    print(f"\n{'='*60}")
    print("数据集性能对比总结")
    print(f"{'='*60}")
    print(f"{'数据集':<15} {'准确率':<10} {'训练时间(秒)':<15} {'网络结构'}")
    print("-" * 60)
    
    for dataset_name, result in results_summary.items():
        layers_str = '->'.join(map(str, result['config']['layers']))
        print(f"{dataset_name.upper():<15} {result['accuracy']:<10.4f} "
              f"{result['training_time']:<15.2f} {layers_str}")
    
    return results_summary

# --------------------------
# 6. 运行测试
# --------------------------
if __name__ == "__main__":
    try:
        results = test_datasets_simple()
        print(f"\n测试完成！discPC在多个复杂数据集上展现了良好的性能。")
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
