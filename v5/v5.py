import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 设置随机种子（确保结果可复现）
torch.manual_seed(0)
np.random.seed(0)

# --------------------------
# 1. 激活函数（适配多类别分类）
# --------------------------
def sigmoid(x):
    # 使用PyTorch的sigmoid，并用clamp防止数值溢出
    return torch.sigmoid(torch.clamp(x, -20, 20))

def softmax(x):
    # 多类别分类输出层使用softmax
    exp_x = torch.exp(torch.clamp(x, -20, 20))  # 防止数值溢出
    return exp_x / torch.sum(exp_x, dim=0, keepdim=True)

# --------------------------
# 2. 数据加载与预处理
# --------------------------
def load_iris_data(test_size=0.2, device='cpu'):
    # 加载IRIS数据集
    iris = load_iris()
    X = iris.data  # 特征 (150, 4)
    y = iris.target.reshape(-1, 1)  # 标签 (150, 1)
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 标签独热编码（适应多类别）
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)  # (150, 3)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y
    )
    
    # 转换为PyTorch张量并移动到目标设备
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device).T  # (4, n_train)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device).T    # (4, n_test)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device).T  # (3, n_train)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device).T    # (3, n_test)
    
    return X_train, X_test, y_train, y_test, iris.target_names

# --------------------------
# 3. 多层discPC网络类（支持多类别）
# --------------------------
class discPC_Flexible:
    def __init__(self, layers, activations=None, use_bias=True, device='cpu'):
        """
        初始化多层discPC网络
        layers: 列表，定义各层神经元数量，例如[4, 8, 3]表示输入层4个，隐藏层8个，输出层3个
        activations: 各层激活函数列表，默认隐藏层用sigmoid，输出层用softmax
        use_bias: 是否使用偏置项
        device: 计算设备（cuda或cpu）
        """
        self.layers = layers  # 层结构定义
        self.n_layers = len(layers)  # 总层数（包括输入层和输出层）
        self.use_bias = use_bias
        self.device = device
        
        # 设置激活函数，默认隐藏层用sigmoid，输出层用softmax
        if activations is None:
            self.activations = [sigmoid] * (self.n_layers - 2) + [softmax]
        else:
            self.activations = activations
        
        # 初始化权重和偏置
        self.weights = []  # 权重列表，weights[i]是连接第i层和第i+1层的权重矩阵
        self.biases = []   # 偏置列表，biases[i]是第i+1层的偏置
        
        # 为每一层初始化权重和偏置
        for i in range(self.n_layers - 1):
            input_dim = layers[i]
            output_dim = layers[i+1]
            
            # 权重初始化：正态分布，方差为1/输入维度
            weight = torch.randn(input_dim, output_dim, device=device) * torch.sqrt(torch.tensor(1/input_dim, device=device))
            self.weights.append(weight)
            
            # 偏置初始化
            if use_bias:
                bias = torch.zeros(output_dim, 1, device=device)
                self.biases.append(bias)
            else:
                self.biases.append(None)
    
    def initialize_activity(self, x_batch):
        """初始化所有层的活动值"""
        activities = [x_batch]  # 第一层活动值是输入
        
        # 逐层计算初始活动值
        for i in range(self.n_layers - 1):
            prev_activity = activities[-1]
            weight = self.weights[i]
            bias = self.biases[i]
            
            # 计算输入
            layer_input = weight.T @ prev_activity
            if self.use_bias and bias is not None:
                layer_input += bias
            
            # 应用对应层的激活函数
            activity = self.activations[i](layer_input)
            activities.append(activity)
        
        return activities
    
    def update_activity(self, x_batch, activities, target=None, lr=0.1, m=20, lambda_target=0.9):
        """更新所有层的活动值"""
        # 复制初始活动值，避免修改原始列表
        current_activities = [act.clone() for act in activities]
        
        for _ in range(m):
            # 前向计算预测值（用于误差计算）
            pred_activities = [x_batch]
            for i in range(self.n_layers - 1):
                prev_pred = pred_activities[-1]
                layer_input = self.weights[i].T @ prev_pred
                if self.use_bias and self.biases[i] is not None:
                    layer_input += self.biases[i]
                pred_activities.append(self.activations[i](layer_input))
            
            # 从输出层反向更新到隐藏层
            for i in reversed(range(1, self.n_layers)):  # i=0是输入层，不更新
                # 计算预测误差
                epsilon = current_activities[i] - pred_activities[i]
                update = -lr * epsilon
                
                # 如果有目标值，且是输出层，添加目标误差
                if target is not None and i == self.n_layers - 1:
                    target_error = current_activities[i] - target
                    update -= lr * lambda_target * target_error
                # 如果有目标值且不是输出层，传播目标误差
                elif target is not None and i < self.n_layers - 1:
                    # 计算激活函数导数（针对sigmoid）
                    sigmoid_deriv = pred_activities[i] * (1 - pred_activities[i])
                    # 从上层传播误差
                    next_error = current_activities[i+1] - target if i+1 == self.n_layers -1 else current_activities[i+1] - pred_activities[i+1]
                    target_error = (self.weights[i] @ next_error) * sigmoid_deriv
                    update -= lr * lambda_target * target_error
                
                # 更新活动值
                current_activities[i] += update
        
        return current_activities
    
    def update_weight(self, x_batch, activities, target, lr=0.01):
        """更新所有层的权重和偏置"""
        # 前向计算预测值
        pred_activities = [x_batch]
        for i in range(self.n_layers - 1):
            prev_pred = pred_activities[-1]
            layer_input = self.weights[i].T @ prev_pred
            if self.use_bias and self.biases[i] is not None:
                layer_input += self.biases[i]
            pred_activities.append(self.activations[i](layer_input))
        
        # 计算各层误差
        errors = [None] * (self.n_layers - 1)  # 误差数量比层数少1
        
        # 输出层误差（多类别）
        errors[-1] = target - pred_activities[-1]
        
        # 反向计算隐藏层误差
        for i in reversed(range(self.n_layers - 2)):  # 从倒数第二层隐藏层开始
            # 假设隐藏层使用sigmoid激活函数
            sigmoid_deriv = pred_activities[i+1] * (1 - pred_activities[i+1])
            errors[i] = (self.weights[i+1] @ errors[i+1]) * sigmoid_deriv
        
        # 更新各层权重和偏置
        batch_size = x_batch.shape[1]
        for i in range(self.n_layers - 1):
            # 更新权重
            self.weights[i] += lr * (pred_activities[i] @ errors[i].T) / batch_size
            
            # 更新偏置
            if self.use_bias and self.biases[i] is not None:
                self.biases[i] += lr * torch.mean(errors[i], dim=1, keepdim=True)
    
    def predict(self, x_batch, m=20):
        """预测函数（多类别）"""
        activities = self.initialize_activity(x_batch)
        activities = self.update_activity(x_batch, activities, m=m)
        # 多类别：取概率最大的类别
        return torch.argmax(activities[-1], dim=0)

# --------------------------
# 4. 测试IRIS数据集
# --------------------------
if __name__ == "__main__":
    # 检测设备（优先GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载并预处理数据
    X_train, X_test, y_train, y_test, class_names = load_iris_data(test_size=0.2, device=device)
    print(f"IRIS数据集加载完成 - 训练样本: {X_train.shape[1]}, 测试样本: {X_test.shape[1]}")
    print(f"类别: {class_names}")
    
    # 超参数配置
    n_epochs = 10000
    # 层结构定义：输入层4个神经元（IRIS有4个特征），隐藏层10个，输出层3个（3个类别）
    layers = [4, 16, 8, 3]
    m = 10
    lr_activity = 0.05
    lr_weight = 0.05
    lambda_target = 0.8
    use_bias = True
    
    # 初始化模型
    model = discPC_Flexible(
        layers=layers,
        use_bias=use_bias,
        device=device
    )
    
    # 训练循环
    print(f"\n开始训练IRIS分类任务（网络结构: {layers}）...")
    for epoch in range(n_epochs):
        # 初始化活动值
        activities = model.initialize_activity(X_train)
        # 更新活动值
        activities = model.update_activity(
            X_train, activities, target=y_train,
            lr=lr_activity, m=m, lambda_target=lambda_target
        )
        
        # 更新权重
        model.update_weight(X_train, activities, y_train, lr=lr_weight)
        
        # 计算训练损失和准确率
        y_pred_train = activities[-1]
        train_loss = torch.mean((y_pred_train - y_train) **2)
        train_pred_label = torch.argmax(y_pred_train, dim=0)
        train_true_label = torch.argmax(y_train, dim=0)
        train_accuracy = torch.mean((train_pred_label == train_true_label).float())
        
        # 每100个epoch打印一次信息
        if (epoch + 1) % 100 == 0:
            # 计算测试准确率
            test_activities = model.initialize_activity(X_test)
            test_activities = model.update_activity(X_test, test_activities, m=m)
            test_pred_label = torch.argmax(test_activities[-1], dim=0)
            test_true_label = torch.argmax(y_test, dim=0)
            test_accuracy = torch.mean((test_pred_label == test_true_label).float())
            
            print(f"Epoch {epoch+1:4d}, 训练损失: {train_loss.item():.6f}, "
                  f"训练准确率: {train_accuracy.item():.4f}, "
                  f"测试准确率: {test_accuracy.item():.4f}")
    
    # 最终测试结果
    print(f"\nIRIS分类任务最终测试结果（网络结构: {layers}）：")
    pred_labels = model.predict(X_test, m=m)
    true_labels = torch.argmax(y_test, dim=0)
    
    # 计算总体准确率
    overall_accuracy = torch.mean((pred_labels == true_labels).float()).item()
    print(f"总体测试准确率: {overall_accuracy:.4f}")
    
    # 计算每个类别的准确率
    for i, class_name in enumerate(class_names):
        mask = (true_labels == i)
        if torch.sum(mask) > 0:
            class_accuracy = torch.mean((pred_labels[mask] == true_labels[mask]).float()).item()
            print(f"{class_name} 准确率: {class_accuracy:.4f}")

    print("\n具体测试样本示例：")
    sample_indices = [0, 5, 10, 15, 20]  # 选取几个测试样本的索引
    for idx in sample_indices:
        x_test_sample = X_test[:, idx:idx+1]  # 单个样本，形状为(4,1)
        true_label = torch.argmax(y_test[:, idx:idx+1], dim=0).item()
        pred_label = model.predict(x_test_sample, m=m).item()
        
        # 还原特征（因为之前做了标准化，这里为了直观展示原始特征量级，可选择是否还原，这里直接展示标准化后的值）
        feature_values = x_test_sample.cpu().numpy().flatten()
        print(f"样本索引 {idx}")
        print(f"输入特征: {feature_values.round(4)}")
        print(f"预测类别: {class_names[pred_label]}")
        print(f"真实类别: {class_names[true_label]}")
        print("-" * 30)