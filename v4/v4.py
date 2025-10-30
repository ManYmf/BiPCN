import torch
# 设置随机种子（确保结果可复现）
torch.manual_seed(0)

# --------------------------
# 1. 激活函数及数据集（适配PyTorch）
# --------------------------
def sigmoid(x):
    # 使用PyTorch的sigmoid，并用clamp防止数值溢出
    return torch.sigmoid(torch.clamp(x, -20, 20))

def generate_data(task='XOR', device='cpu'):
    # 生成数据并转换为PyTorch张量，直接移到目标设备（GPU/CPU）
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=device)
    if task == 'AND':
        y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32, device=device)
    elif task == 'OR':
        y = torch.tensor([[0], [1], [1], [1]], dtype=torch.float32, device=device)
    elif task == 'XOR':
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32, device=device)
    elif task == 'NAND':
        y = torch.tensor([[1], [1], [1], [0]], dtype=torch.float32, device=device)
    elif task == 'NOR':
        y = torch.tensor([[1], [0], [0], [0]], dtype=torch.float32, device=device)
    else:
        raise ValueError("任务必须是'AND'、'OR'、'XOR'、'NAND'或'NOR'")
    return X, y

# --------------------------
# 2. 通用多层discPC网络类（可灵活定义层数和神经元数）
# --------------------------
class discPC_Flexible:
    def __init__(self, layers, activation=sigmoid, use_bias=True, device='cpu'):
        """
        初始化多层discPC网络
        layers: 列表，定义各层神经元数量，例如[2, 3, 1]表示输入层2个神经元，隐藏层3个，输出层1个
        activation: 激活函数
        use_bias: 是否使用偏置项
        device: 计算设备（cuda或cpu）
        """
        self.layers = layers  # 层结构定义
        self.n_layers = len(layers)  # 总层数（包括输入层和输出层）
        self.activation = activation
        self.use_bias = use_bias
        self.device = device
        
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
            
            # 应用激活函数
            activity = self.activation(layer_input)
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
                pred_activities.append(self.activation(layer_input))
            
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
                    # 计算激活函数导数
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
            pred_activities.append(self.activation(layer_input))
        
        # 计算各层误差
        errors = [None] * (self.n_layers - 1)  # 误差数量比层数少1
        
        # 输出层误差
        errors[-1] = target - pred_activities[-1]
        
        # 反向计算隐藏层误差
        for i in reversed(range(self.n_layers - 2)):  # 从倒数第二层隐藏层开始
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
        """预测函数"""
        activities = self.initialize_activity(x_batch)
        activities = self.update_activity(x_batch, activities, m=m)
        # 输出层大于0.5视为1，否则为0
        return (activities[-1] > 0.5).float()

# --------------------------
# 3. 测试灵活多层结构
# --------------------------
if __name__ == "__main__":
    # 检测设备（优先GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 超参数配置
    task = 'XOR'
    n_epochs = 10000
    # 层结构定义：输入层2个神经元，两个隐藏层分别为4和2个神经元，输出层1个神经元
    layers = [2, 4, 2, 1]  # 可根据需要修改
    m = 8
    lr_activity = 0.1
    lr_weight = 0.1
    lambda_target = 0.9
    use_bias = True
    
    # 生成数据并转换为批格式
    X, y = generate_data(task, device=device)
    X_batch = X.T  # (input_dim, batch_size)
    y_batch = y.T  # (output_dim, batch_size)
    
    # 初始化模型
    model = discPC_Flexible(
        layers=layers,
        use_bias=use_bias,
        device=device
    )
    
    # 训练循环
    print(f"开始训练{task}任务（网络结构: {layers}，偏置项{'启用' if use_bias else '禁用'}）...")
    for epoch in range(n_epochs):
        # 初始化活动值
        activities = model.initialize_activity(X_batch)
        # 更新活动值
        activities = model.update_activity(
            X_batch, activities, target=y_batch,
            lr=lr_activity, m=m, lambda_target=lambda_target
        )
        
        # 更新权重
        model.update_weight(X_batch, activities, y_batch, lr=lr_weight)
        
        # 计算损失和准确率
        y_pred = activities[-1]
        loss = torch.mean((y_pred - y_batch) **2)
        pred_label = (y_pred > 0.5).float()
        accuracy = torch.mean((pred_label == y_batch).float())
        
        # 打印进度
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}, 损失: {loss.item():.4f}, 准确率: {accuracy.item():.4f}")
    
    # 测试结果
    print(f"\n{task}任务测试结果（网络结构: {layers}）：")
    pred_batch = model.predict(X_batch, m=m)
    correct_count = 0
    for i in range(X.shape[0]):
        x = X[i].cpu().numpy()
        pred = pred_batch[0, i].item()
        true = y[i, 0].item()
        is_correct = (pred == true)
        correct_count += 1 if is_correct else 0
        print(f"输入: {x}, 预测: {pred}, 真实: {true}, {'✓' if is_correct else '✗'}")
    print(f"\n最终测试准确率: {correct_count/X.shape[0]:.4f}")