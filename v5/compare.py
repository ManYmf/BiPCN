import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 设置随机种子（确保结果可复现）
torch.manual_seed(0)
np.random.seed(0)

# --------------------------
# 1. 数据加载与预处理
# --------------------------
def load_iris_data(test_size=0.2, device='cpu'):
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)
    
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 标签独热编码
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y
    )
    
    # 转换为PyTorch张量并移动到目标设备
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
    
    return X_train, X_test, y_train, y_test, iris.target_names

# --------------------------
# 2. 全连接反向传播神经网络类
# --------------------------
class FullyConnectedNN(nn.Module):
    def __init__(self, layers=[4, 16, 8, 3]):
        super(FullyConnectedNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        # 输出层使用Softmax（多分类）
        self.softmax = nn.Softmax(dim=1)
        # 隐藏层使用Sigmoid（与discPC保持激活函数一致，便于对比）
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.sigmoid(x)
        # 输出层
        x = self.layers[-1](x)
        x = self.softmax(x)
        return x

# --------------------------
# 3. 训练与测试流程
# --------------------------
if __name__ == "__main__":
    # 检测设备（优先GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载并预处理数据
    X_train, X_test, y_train, y_test, class_names = load_iris_data(test_size=0.2, device=device)
    print(f"IRIS数据集加载完成 - 训练样本: {X_train.shape[0]}, 测试样本: {X_test.shape[0]}")
    print(f"类别: {class_names}")
    
    # 初始化模型、损失函数和优化器
    model = FullyConnectedNN(layers=[4, 16, 8, 3]).to(device)
    criterion = nn.MSELoss()  # 与discPC的均方误差保持一致
    optimizer = optim.SGD(model.parameters(), lr=0.05)  # 学习率与discPC对齐
    
    # 超参数配置
    n_epochs = 5000
    
    # 训练循环
    print(f"\n开始训练全连接反向传播神经网络（结构: [4, 16, 8, 3]）...")
    for epoch in range(n_epochs):
        # 前向传播
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        # 反向传播与参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算训练准确率
        train_pred_label = torch.argmax(y_pred, dim=1)
        train_true_label = torch.argmax(y_train, dim=1)
        train_accuracy = torch.mean((train_pred_label == train_true_label).float())
        
        # 每100个epoch打印一次信息
        if (epoch + 1) % 100 == 0:
            # 测试集评估
            with torch.no_grad():
                y_pred_test = model(X_test)
                test_pred_label = torch.argmax(y_pred_test, dim=1)
                test_true_label = torch.argmax(y_test, dim=1)
                test_accuracy = torch.mean((test_pred_label == test_true_label).float())
            
            print(f"Epoch {epoch+1:4d}, 训练损失: {loss.item():.6f}, "
                  f"训练准确率: {train_accuracy.item():.4f}, "
                  f"测试准确率: {test_accuracy.item():.4f}")
    
    # 最终测试结果
    print(f"\n全连接网络最终测试结果（结构: [4, 16, 8, 3]）：")
    with torch.no_grad():
        y_pred_test = model(X_test)
        test_pred_label = torch.argmax(y_pred_test, dim=1)
        test_true_label = torch.argmax(y_test, dim=1)
        overall_accuracy = torch.mean((test_pred_label == test_true_label).float()).item()
        print(f"总体测试准确率: {overall_accuracy:.4f}")
        
        # 分类别准确率
        for i, class_name in enumerate(class_names):
            mask = (test_true_label == i)
            if torch.sum(mask) > 0:
                class_accuracy = torch.mean((test_pred_label[mask] == test_true_label[mask]).float()).item()
                print(f"{class_name} 准确率: {class_accuracy:.4f}")