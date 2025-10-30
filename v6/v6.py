import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader

# 设置随机种子（确保结果可复现）
torch.manual_seed(0)
np.random.seed(0)

# --------------------------
# 1. 激活函数（保持原有，适配多类别）
# --------------------------
def sigmoid(x):
    return torch.sigmoid(torch.clamp(x, -20, 20))

def softmax(x):
    exp_x = torch.exp(torch.clamp(x, -20, 20))
    return exp_x / torch.sum(exp_x, dim=0, keepdim=True)

# --------------------------
# 2. 数据加载与预处理（适配MNIST）
# --------------------------
def load_mnist_data(batch_size=64, device='cpu'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值和标准差
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    encoder.fit(np.arange(10).reshape(-1, 1))
    
    return train_loader, test_loader, encoder, device

# --------------------------
# 3. 优化后的多层discPC网络类（贴合变分同步更新）
# --------------------------
class discPC_Flexible:
    def __init__(self, layers, activations=None, use_bias=True, device='cpu'):
        self.layers = layers
        self.n_layers = len(layers)
        self.use_bias = use_bias
        self.device = device
        
        # 激活函数默认配置：隐藏层sigmoid，输出层softmax
        if activations is None:
            self.activations = [sigmoid] * (self.n_layers - 2) + [softmax]
        else:
            self.activations = activations
        
        # 初始化权重和偏置（Xavier初始化）
        self.weights = []
        self.biases = []
        for i in range(self.n_layers - 1):
            input_dim = layers[i]
            output_dim = layers[i+1]
            weight = torch.randn(input_dim, output_dim, device=device) * torch.sqrt(torch.tensor(1/input_dim, device=device))
            self.weights.append(weight)
            self.biases.append(torch.zeros(output_dim, 1, device=device) if use_bias else None)
    
    # 封装：前向计算所有预测活动值（一次性得到所有层的预测）
    def _compute_pred_activities(self, x_batch):
        pred_activities = [x_batch]
        for i in range(self.n_layers - 1):
            prev_pred = pred_activities[i]
            layer_input = self.weights[i].T @ prev_pred
            if self.use_bias and self.biases[i] is not None:
                layer_input += self.biases[i]
            pred_activities.append(self.activations[i](layer_input))
        return pred_activities
    
    # 封装：计算所有层的误差（顺序依赖，但一次性返回完整误差列表）
    def _compute_errors(self, pred_activities, target):
        errors = [None] * (self.n_layers - 1)
        # 输出层误差
        errors[-1] = target - pred_activities[-1]
        # 隐藏层误差（反向传播，基于下一层误差）
        for i in reversed(range(self.n_layers - 2)):
            sigmoid_deriv = pred_activities[i+1] * (1 - pred_activities[i+1])
            errors[i] = (self.weights[i+1] @ errors[i+1]) * sigmoid_deriv
        return errors
    
    # 活动值初始化（与原逻辑一致）
    def initialize_activity(self, x_batch):
        return self._compute_pred_activities(x_batch)  # 直接复用前向计算方法
    
    # 活动值更新（同步更新：基于同一轮预测活动值更新所有层）
    def update_activity(self, x_batch, activities, target=None, lr=0.1, m=20, lambda_target=0.9):
        current_activities = [act.clone() for act in activities]
        for _ in range(m):
            # 1. 一次性计算所有层的预测活动值（固定当前权重）
            pred_activities = self._compute_pred_activities(x_batch)
            
            # 2. 反向更新所有层的活动值（基于同一pred_activities，同步更新）
            for i in reversed(range(1, self.n_layers)):
                # 基础误差：当前活动值 - 预测活动值
                epsilon = current_activities[i] - pred_activities[i]
                update = -lr * epsilon
                
                # 加入目标误差（输出层直接用目标，隐藏层反向传播）
                if target is not None:
                    if i == self.n_layers - 1:
                        target_error = current_activities[i] - target
                    else:
                        sigmoid_deriv = pred_activities[i] * (1 - pred_activities[i])
                        next_error = current_activities[i+1] - target if (i+1 == self.n_layers -1) else (current_activities[i+1] - pred_activities[i+1])
                        target_error = (self.weights[i] @ next_error) * sigmoid_deriv
                    update -= lr * lambda_target * target_error
                
                current_activities[i] += update
        return current_activities
    
    # 权重更新（同步更新：基于完整误差列表更新所有权重）
    def update_weight(self, x_batch, activities, target, lr=0.01):
        # 1. 一次性计算所有预测活动值和误差（固定当前活动值）
        pred_activities = self._compute_pred_activities(x_batch)
        errors = self._compute_errors(pred_activities, target)  # 得到完整误差列表
        
        # 2. 同步更新所有权重和偏置（基于同一套误差）
        batch_size = x_batch.shape[1]
        for i in range(self.n_layers - 1):
            # 权重更新：输入活动值 × 误差的转置（批量平均）
            self.weights[i] += lr * (pred_activities[i] @ errors[i].T) / batch_size
            # 偏置更新：误差的均值
            if self.use_bias and self.biases[i] is not None:
                self.biases[i] += lr * torch.mean(errors[i], dim=1, keepdim=True)
    
    # 预测方法（保持不变）
    def predict(self, x_batch, m=20):
        activities = self.initialize_activity(x_batch)
        activities = self.update_activity(x_batch, activities, m=m)
        return torch.argmax(activities[-1], dim=0)

# --------------------------
# 4. 测试MNIST数据集（核心训练逻辑不变）
# --------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    batch_size = 64
    train_loader, test_loader, encoder, device = load_mnist_data(batch_size=batch_size, device=device)
    print(f"MNIST数据集加载完成 - 训练批次: {len(train_loader)}, 测试批次: {len(test_loader)}")
    print(f"每个批次大小: {batch_size}, 类别数: 10（数字0-9）")
    
    # 超参数配置
    n_epochs = 50
    layers = [784, 128, 64, 10]
    m = 15
    lr_activity = 0.03
    lr_weight = 0.03
    lambda_target = 0.7
    use_bias = True
    
    model = discPC_Flexible(
        layers=layers,
        use_bias=use_bias,
        device=device
    )
    
    # 训练循环
    print(f"\n开始训练MNIST分类任务（网络结构: {layers}）...")
    for epoch in range(n_epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            x_batch = data.view(-1, 784).T.to(device)
            y_batch = encoder.transform(targets.cpu().numpy().reshape(-1, 1))
            y_batch = torch.tensor(y_batch, dtype=torch.float32, device=device).T
        
            # 核心训练步骤（逻辑不变，调用优化后的方法）
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
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # 测试集评估
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                x_batch = data.view(-1, 784).T.to(device)
                y_batch = encoder.transform(targets.cpu().numpy().reshape(-1, 1))
                y_batch = torch.tensor(y_batch, dtype=torch.float32, device=device).T
                
                test_activities = model.initialize_activity(x_batch)
                test_activities = model.update_activity(x_batch, test_activities, m=m)
                pred_labels = torch.argmax(test_activities[-1], dim=0)
                true_labels = torch.argmax(y_batch, dim=0)
                
                test_correct += torch.sum(pred_labels == true_labels).item()
                test_total += targets.size(0)
        
        test_accuracy = test_correct / test_total
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}, 训练损失: {avg_train_loss:.6f}, "
                  f"训练准确率: {train_accuracy:.4f}, "
                  f"测试准确率: {test_accuracy:.4f}")
    
    # 最终测试结果
    print(f"\nMNIST分类任务最终测试结果（网络结构: {layers}）：")
    print(f"总体测试准确率: {test_accuracy:.4f}")
    
    # 每个类别的准确率
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data, targets in test_loader:
            x_batch = data.view(-1, 784).T.to(device)
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
    
    for i in range(10):
        print(f"数字 {i} 准确率: {class_correct[i]/class_total[i]:.4f}")

    # 示例预测
    print("\n具体测试样本示例：")
    test_iter = iter(test_loader)
    data, targets = next(test_iter)
    x_batch = data.view(-1, 784).T.to(device)
    y_batch = encoder.transform(targets.cpu().numpy().reshape(-1, 1))
    y_batch = torch.tensor(y_batch, dtype=torch.float32, device=device).T
    
    pred_labels = model.predict(x_batch, m=m)
    true_labels = torch.argmax(y_batch, dim=0)
    
    for idx in range(5):
        print(f"样本 {idx+1}")
        print(f"预测数字: {pred_labels[idx].item()}, 真实数字: {true_labels[idx].item()}")
        print("-" * 30)