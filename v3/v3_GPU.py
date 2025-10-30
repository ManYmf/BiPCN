import torch
# 设置随机种子（确保结果可复现）
torch.manual_seed(1)

# --------------------------
# 1. 激活函数及数据集（适配PyTorch）
# --------------------------
def sigmoid(x):
    # 使用PyTorch的sigmoid，并用clamp防止数值溢出（同原逻辑）
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
# 2. 多层discPC网络类（GPU加速版）
# --------------------------
class discPC_MultiLayer:
    def __init__(self, input_dim, hidden_dim, output_dim, activation=sigmoid, use_bias=True, device='cpu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.device = device  # 设备（cuda或cpu）
        
        # 权重初始化（PyTorch张量，直接存到GPU）
        # 正态分布初始化，同原逻辑但用torch.randn
        self.V1 = torch.randn(input_dim, hidden_dim, device=device) * torch.sqrt(torch.tensor(1/input_dim, device=device))
        self.V2 = torch.randn(hidden_dim, output_dim, device=device) * torch.sqrt(torch.tensor(1/hidden_dim, device=device))
        
        # 偏置（存到GPU）
        self.b1 = torch.zeros(hidden_dim, 1, device=device) if use_bias else None
        self.b2 = torch.zeros(output_dim, 1, device=device) if use_bias else None
    
    def initialize_activity(self, x_batch):
        # x_batch: (input_dim, batch_size)，已在GPU上
        h_input = self.V1.T @ x_batch  # 矩阵乘法（GPU加速）
        if self.use_bias:
            h_input += self.b1  # 广播机制（同NumPy，GPU支持）
        h = self.activation(h_input)
        
        y_input = self.V2.T @ h
        if self.use_bias:
            y_input += self.b2
        y = self.activation(y_input)
        
        return h, y
    
    def update_activity(self, x_batch, h, y, target=None, lr=0.1, m=20, lambda_target=0.9):
        # 所有变量均为GPU张量
        for _ in range(m):
            h_input = self.V1.T @ x_batch
            if self.use_bias:
                h_input += self.b1
            h_pred = self.activation(h_input)
            
            y_input = self.V2.T @ h_pred
            if self.use_bias:
                y_input += self.b2
            y_pred = self.activation(y_input)
            
            # 输出层活动更新（GPU上并行计算）
            epsilon_y = y - y_pred
            update_y = -lr * epsilon_y
            if target is not None:
                target_error_y = y - target
                update_y -= lr * lambda_target * target_error_y
            y = y + update_y
            
            # 隐藏层活动更新（GPU上并行计算）
            epsilon_h = h - h_pred
            update_h = -lr * epsilon_h
            if target is not None:
                sigmoid_deriv_h = h_pred * (1 - h_pred)
                target_error_h = (self.V2 @ target_error_y) * sigmoid_deriv_h
                update_h -= lr * lambda_target * target_error_h
            h = h + update_h
        
        return h, y
    
    def update_weight(self, x_batch, h, y, target, lr=0.01):
        # 权重更新（GPU上并行计算）
        h_input = self.V1.T @ x_batch
        if self.use_bias:
            h_input += self.b1
        h_pred = self.activation(h_input)
        
        y_input = self.V2.T @ h_pred
        if self.use_bias:
            y_input += self.b2
        y_pred = self.activation(y_input)
        
        # 批量误差计算
        error_y = target - y_pred
        sigmoid_deriv_h = h_pred * (1 - h_pred)
        error_h = (self.V2 @ error_y) * sigmoid_deriv_h
        
        # 权重更新（平均批量梯度，GPU加速）
        batch_size = x_batch.shape[1]
        self.V2 += lr * (h_pred @ error_y.T) / batch_size
        self.V1 += lr * (x_batch @ error_h.T) / batch_size
        
        # 偏置更新（批量平均）
        if self.use_bias:
            self.b2 += lr * torch.mean(error_y, dim=1, keepdim=True)  # keepdim对应NumPy的keepdims
            self.b1 += lr * torch.mean(error_h, dim=1, keepdim=True)
    
    def predict(self, x_batch, m=20):
        h, y = self.initialize_activity(x_batch)
        h, y = self.update_activity(x_batch, h, y, m=m)
        return (y > 0.5).float()  # 转为0/1预测

# --------------------------
# 3. GPU加速训练与测试
# --------------------------
if __name__ == "__main__":
    # 检测设备（优先GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")  # 输出当前设备（cuda或cpu）
    
    # 超参数配置
    task = 'XOR'
    n_epochs = 5000
    hidden_dim = 2
    m = 50
    lr_activity = 0.1
    lr_weight = 0.1
    lambda_target = 0.9
    use_bias = True
    
    # 生成数据并转换为批格式（已移到GPU）
    X, y = generate_data(task, device=device)  # X: (4,2)，y: (4,1)
    X_batch = X.T  # (input_dim, batch_size) → (2,4)（列是样本）
    y_batch = y.T  # (output_dim, batch_size) → (1,4)
    input_dim, output_dim = X.shape[1], y.shape[1]
    
    # 初始化模型（参数直接存到GPU）
    model = discPC_MultiLayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_bias=use_bias,
        device=device
    )
    
    # 训练循环（GPU加速）
    print(f"开始训练{task}任务（批处理+GPU加速，偏置项{'启用' if use_bias else '禁用'}）...")
    for epoch in range(n_epochs):
        # 批量处理（所有计算在GPU上）
        h, y_pred = model.initialize_activity(X_batch)
        h, y_pred = model.update_activity(
            X_batch, h, y_pred, target=y_batch,
            lr=lr_activity, m=m, lambda_target=lambda_target
        )
        
        # 更新权重（GPU上计算）
        model.update_weight(X_batch, h, y_pred, y_batch, lr=lr_weight)
        
        # 计算损失和准确率（GPU上直接计算）
        loss = torch.mean((y_pred - y_batch)** 2)  # 批量MSE
        pred_label = (y_pred > 0.5).float()
        accuracy = torch.mean((pred_label == y_batch).float())  # 批量准确率
        
        # 打印进度
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}, 损失: {loss.item():.4f}, 准确率: {accuracy.item():.4f}")
    
    # 测试结果（GPU计算后，结果可直接用.item()取 scalar）
    print(f"\n{task}任务测试结果（批处理+GPU加速）：")
    pred_batch = model.predict(X_batch, m=m)  # (1,4)，GPU上的张量
    correct_count = 0
    for i in range(X.shape[0]):
        x = X[i].cpu().numpy()  # 移到CPU并转NumPy（仅为打印）
        pred = pred_batch[0, i].item()  # 取标量值
        true = y[i, 0].item()
        is_correct = (pred == true)
        correct_count += 1 if is_correct else 0
        print(f"输入: {x}, 预测: {pred}, 真实: {true}, {'✓' if is_correct else '✗'}")
    print(f"\n最终测试准确率: {correct_count/X.shape[0]:.4f}")