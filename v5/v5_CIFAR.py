import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

# --------------------------
# 1. 激活函数（保持不变）
# --------------------------
def sigmoid(x):
    return torch.sigmoid(torch.clamp(x, -20, 20))

def softmax(x):
    exp_x = torch.exp(torch.clamp(x, -20, 20))
    return exp_x / torch.sum(exp_x, dim=0, keepdim=True)

# --------------------------
# 2. 数据加载与预处理（适配CIFAR-10）
# --------------------------
def load_cifar10_data(batch_size=128, device='cpu'):
    # 数据增强（针对训练集，提升泛化能力）
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),     # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(                 # CIFAR-10均值和标准差
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    # 测试集仅做标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    # 加载CIFAR-10数据集
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 类别名称（CIFAR-10的10个类别）
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 独热编码器
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    encoder.fit(np.arange(10).reshape(-1, 1))
    
    return train_loader, test_loader, encoder, class_names, device

# --------------------------
# 3. 多层discPC网络类（保持核心逻辑）
# --------------------------
class discPC_Flexible:
    def __init__(self, layers, activations=None, use_bias=True, device='cpu'):
        self.layers = layers
        self.n_layers = len(layers)
        self.use_bias = use_bias
        self.device = device
        
        if activations is None:
            self.activations = [sigmoid] * (self.n_layers - 2) + [softmax]
        else:
            self.activations = activations
        
        # 初始化权重和偏置（适配高维输入）
        self.weights = []
        self.biases = []
        for i in range(self.n_layers - 1):
            input_dim = layers[i]
            output_dim = layers[i+1]
            # 权重初始化：针对高维输入，缩小初始方差
            weight = torch.randn(input_dim, output_dim, device=device) * torch.sqrt(torch.tensor(1/(input_dim*2), device=device))
            self.weights.append(weight)
            if use_bias:
                self.biases.append(torch.zeros(output_dim, 1, device=device))
            else:
                self.biases.append(None)
    
    def initialize_activity(self, x_batch):
        activities = [x_batch]
        for i in range(self.n_layers - 1):
            prev_activity = activities[-1]
            layer_input = self.weights[i].T @ prev_activity
            if self.use_bias and self.biases[i] is not None:
                layer_input += self.biases[i]
            activity = self.activations[i](layer_input)
            activities.append(activity)
        return activities
    
    def update_activity(self, x_batch, activities, target=None, lr=0.1, m=20, lambda_target=0.9):
        current_activities = [act.clone() for act in activities]
        for _ in range(m):
            # 前向计算预测值
            pred_activities = [x_batch]
            for i in range(self.n_layers - 1):
                prev_pred = pred_activities[-1]
                layer_input = self.weights[i].T @ prev_pred
                if self.use_bias and self.biases[i] is not None:
                    layer_input += self.biases[i]
                pred_activities.append(self.activations[i](layer_input))
            
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
        # 前向计算预测值
        pred_activities = [x_batch]
        for i in range(self.n_layers - 1):
            prev_pred = pred_activities[-1]
            layer_input = self.weights[i].T @ prev_pred
            if self.use_bias and self.biases[i] is not None:
                layer_input += self.biases[i]
            pred_activities.append(self.activations[i](layer_input))
        
        # 计算误差
        errors = [None] * (self.n_layers - 1)
        errors[-1] = target - pred_activities[-1]
        for i in reversed(range(self.n_layers - 2)):
            sigmoid_deriv = pred_activities[i+1] * (1 - pred_activities[i+1])
            errors[i] = (self.weights[i+1] @ errors[i+1]) * sigmoid_deriv
        
        # 更新权重和偏置（增加小批量平均）
        batch_size = x_batch.shape[1]
        for i in range(self.n_layers - 1):
            self.weights[i] += lr * (pred_activities[i] @ errors[i].T) / batch_size
            if self.use_bias and self.biases[i] is not None:
                self.biases[i] += lr * torch.mean(errors[i], dim=1, keepdim=True)
    
    def predict(self, x_batch, m=20):
        activities = self.initialize_activity(x_batch)
        activities = self.update_activity(x_batch, activities, m=m)
        return torch.argmax(activities[-1], dim=0)

# --------------------------
# 4. 测试CIFAR-10数据集
# --------------------------
if __name__ == "__main__":
    # 设备配置（CIFAR训练建议用GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cpu':
        print("警告：CIFAR-10训练耗时较长，建议使用GPU加速")
    
    # 加载数据
    batch_size = 128
    train_loader, test_loader, encoder, class_names, device = load_cifar10_data(batch_size=batch_size, device=device)
    print(f"CIFAR-10数据集加载完成 - 训练批次: {len(train_loader)}, 测试批次: {len(test_loader)}")
    print(f"每个批次大小: {batch_size}, 类别: {class_names}")
    
    # 超参数配置（适配CIFAR的高复杂度）
    n_epochs = 150  # 更多训练轮次
    layers = [3072, 1024, 512, 256, 10]  # 3072=32×32×3，逐步降维至10类
    m = 8  # 活动值更新次数不变
    lr_activity = 0.02  # 降低学习率避免过拟合
    lr_weight = 0.02
    lambda_target = 0.6  # 调整目标误差权重
    use_bias = True
    
    # 初始化模型
    model = discPC_Flexible(
        layers=layers,
        use_bias=use_bias,
        device=device
    )
    
    # 训练循环
    print(f"\n开始训练CIFAR-10分类任务（网络结构: {layers}）...")
    for epoch in range(n_epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 批量训练
        for batch_idx, (data, targets) in enumerate(train_loader):
            # 数据形状调整：(batch_size, 3, 32, 32) → (3072, batch_size)
            x_batch = data.view(-1, 3072).T.to(device)  # 展平为3072维特征
            # 标签独热编码：(batch_size,) → (10, batch_size)
            y_batch = encoder.transform(targets.cpu().numpy().reshape(-1, 1))
            y_batch = torch.tensor(y_batch, dtype=torch.float32, device=device).T
        
            # 训练步骤
            activities = model.initialize_activity(x_batch)
            activities = model.update_activity(
                x_batch, activities, target=y_batch,
                lr=lr_activity, m=m, lambda_target=lambda_target
            )
            model.update_weight(x_batch, activities, y_batch, lr=lr_weight)
        
            # 计算训练指标
            y_pred = activities[-1]
            batch_loss = torch.mean((y_pred - y_batch) ** 2)
            train_loss += batch_loss.item()
            pred_labels = torch.argmax(y_pred, dim=0)
            true_labels = torch.argmax(y_batch, dim=0)
            train_correct += torch.sum(pred_labels == true_labels).item()
            train_total += targets.size(0)
        
        #  epoch级训练指标
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # 测试集评估
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                x_batch = data.view(-1, 3072).T.to(device)
                y_batch = encoder.transform(targets.cpu().numpy().reshape(-1, 1))
                y_batch = torch.tensor(y_batch, dtype=torch.float32, device=device).T
                
                # 预测
                test_activities = model.initialize_activity(x_batch)
                test_activities = model.update_activity(x_batch, test_activities, m=m)
                pred_labels = torch.argmax(test_activities[-1], dim=0)
                true_labels = torch.argmax(y_batch, dim=0)
                
                test_correct += torch.sum(pred_labels == true_labels).item()
                test_total += targets.size(0)
        
        test_accuracy = test_correct / test_total
        
        # 打印日志（每10个epoch）
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}, 训练损失: {avg_train_loss:.6f}, "
                  f"训练准确率: {train_accuracy:.4f}, "
                  f"测试准确率: {test_accuracy:.4f}")
    
    # 最终结果
    print(f"\nCIFAR-10分类任务最终测试结果（网络结构: {layers}）：")
    print(f"总体测试准确率: {test_accuracy:.4f}")
    
    # 每个类别的准确率
    class_correct = list(0. for _ in class_names)
    class_total = list(0. for _ in class_names)
    with torch.no_grad():
        for data, targets in test_loader:
            x_batch = data.view(-1, 3072).T.to(device)
            y_batch = encoder.transform(targets.cpu().numpy().reshape(-1, 1))
            y_batch = torch.tensor(y_batch, dtype=torch.float32, device=device).T
            
            test_activities = model.initialize_activity(x_batch)
            test_activities = model.update_activity(x_batch, test_activities, m=m)
            pred_labels = torch.argmax(test_activities[-1], dim=0)
            true_labels = torch.argmax(y_batch, dim=0)
            
            c = (pred_labels == true_labels).squeeze()
            for i in range(len(targets)):
                label = true_labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name} 准确率: {class_correct[i]/class_total[i]:.4f}")

    # 示例预测
    print("\n具体测试样本示例：")
    test_iter = iter(test_loader)
    data, targets = next(test_iter)
    x_batch = data.view(-1, 3072).T.to(device)
    y_batch = encoder.transform(targets.cpu().numpy().reshape(-1, 1))
    y_batch = torch.tensor(y_batch, dtype=torch.float32, device=device).T
    
    pred_labels = model.predict(x_batch, m=m)
    true_labels = torch.argmax(y_batch, dim=0)
    
    for idx in range(5):
        print(f"样本 {idx+1}")
        print(f"预测类别: {class_names[pred_labels[idx].item()]}")
        print(f"真实类别: {class_names[true_labels[idx].item()]}")
        print("-" * 50)