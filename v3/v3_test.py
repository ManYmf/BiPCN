import numpy as np

# --------------------------
# 1. 激活函数及数据集（不变）
# --------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def generate_data(task='XOR'):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    if task == 'AND':
        y = np.array([[0], [0], [0], [1]], dtype=np.float32)
    elif task == 'OR':
        y = np.array([[0], [1], [1], [1]], dtype=np.float32)
    elif task == 'XOR':
        y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    elif task == 'NAND':
        y = np.array([[1], [1], [1], [0]], dtype=np.float32)
    elif task == 'NOR':
        y = np.array([[1], [0], [0], [0]], dtype=np.float32)
    else:
        raise ValueError("任务必须是'AND'、'OR'、'XOR'、'NAND'或'NOR'")
    return X, y

# --------------------------
# 2. 多层discPC网络类（适配批处理）
# --------------------------
class discPC_MultiLayer:
    def __init__(self, input_dim, hidden_dim, output_dim, activation=sigmoid, use_bias=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        
        # 权重初始化（维度不变）
        self.V1 = np.random.normal(0, np.sqrt(1 / self.input_dim), (self.input_dim, self.hidden_dim))
        self.V2 = np.random.normal(0, np.sqrt(1 / self.hidden_dim), (self.hidden_dim, self.output_dim))
        
        # 偏置（保持维度，后续通过广播适配批处理）
        self.b1 = np.zeros((hidden_dim, 1)) if use_bias else None
        self.b2 = np.zeros((output_dim, 1)) if use_bias else None
    
    def initialize_activity(self, x_batch):
        # x_batch: 输入批量，维度为 (input_dim, batch_size)
        h_input = self.V1.T @ x_batch  # (hidden_dim, input_dim) @ (input_dim, batch_size) → (hidden_dim, batch_size)
        if self.use_bias:
            h_input += self.b1  # 广播偏置 (hidden_dim, 1) → (hidden_dim, batch_size)
        h = self.activation(h_input)  # (hidden_dim, batch_size)
        
        y_input = self.V2.T @ h  # (output_dim, hidden_dim) @ (hidden_dim, batch_size) → (output_dim, batch_size)
        if self.use_bias:
            y_input += self.b2  # 广播偏置 (output_dim, 1) → (output_dim, batch_size)
        y = self.activation(y_input)  # (output_dim, batch_size)
        
        return h, y
    
    def update_activity(self, x_batch, h, y, target=None, lr=0.1, m=20, lambda_target=0.9):
        # x_batch: (input_dim, batch_size)
        # h: (hidden_dim, batch_size)，y: (output_dim, batch_size)
        # target: (output_dim, batch_size)（批量目标）
        for _ in range(m):
            # 计算预测值（批量）
            h_input = self.V1.T @ x_batch
            if self.use_bias:
                h_input += self.b1
            h_pred = self.activation(h_input)  # (hidden_dim, batch_size)
            
            y_input = self.V2.T @ h_pred
            if self.use_bias:
                y_input += self.b2
            y_pred = self.activation(y_input)  # (output_dim, batch_size)
            
            # 更新输出层活动值
            epsilon_y = y - y_pred  # (output_dim, batch_size)
            update_y = -lr * epsilon_y
            if target is not None:
                target_error_y = y - target  # (output_dim, batch_size)
                update_y -= lr * lambda_target * target_error_y
            y = y + update_y  # 批量更新
            
            # 更新隐藏层活动值
            epsilon_h = h - h_pred  # (hidden_dim, batch_size)
            update_h = -lr * epsilon_h
            if target is not None:
                sigmoid_deriv_h = h_pred * (1 - h_pred)  # (hidden_dim, batch_size)
                # 目标误差反向传播到隐藏层（批量）
                target_error_h = (self.V2 @ target_error_y) * sigmoid_deriv_h  # (hidden_dim, batch_size)
                update_h -= lr * lambda_target * target_error_h
            h = h + update_h  # 批量更新
        
        return h, y
    
    def update_weight(self, x_batch, h, y, target, lr=0.01):
        # 批量更新权重（基于整个批次的误差）
        h_input = self.V1.T @ x_batch
        if self.use_bias:
            h_input += self.b1
        h_pred = self.activation(h_input)  # (hidden_dim, batch_size)
        
        y_input = self.V2.T @ h_pred
        if self.use_bias:
            y_input += self.b2
        y_pred = self.activation(y_input)  # (output_dim, batch_size)
        
        # 计算批量误差
        error_y = target - y_pred  # (output_dim, batch_size)
        sigmoid_deriv_h = h_pred * (1 - h_pred)  # (hidden_dim, batch_size)
        error_h = (self.V2 @ error_y) * sigmoid_deriv_h  # (hidden_dim, batch_size)
        
        # 权重更新（批量梯度平均）
        batch_size = x_batch.shape[1]  # 批次大小（这里是4）
        self.V2 += lr * (h_pred @ error_y.T) / batch_size  # 平均每个样本的贡献
        self.V1 += lr * (x_batch @ error_h.T) / batch_size  # 平均每个样本的贡献
        
        # 偏置更新（批量平均）
        if self.use_bias:
            self.b2 += lr * np.mean(error_y, axis=1, keepdims=True)  # 按批次平均
            self.b1 += lr * np.mean(error_h, axis=1, keepdims=True)  # 按批次平均
    
    def predict(self, x_batch, m=20):
        # 批量预测
        h, y = self.initialize_activity(x_batch)
        h, y = self.update_activity(x_batch, h, y, m=m)
        return (y > 0.5).astype(float)  # (output_dim, batch_size)

# --------------------------
# 3. 随机种子遍历测试
# --------------------------
def test_seed(seed, task='XOR', n_epochs=5000, hidden_dim=2, m=50, 
              lr_activity=0.1, lr_weight=0.1, lambda_target=0.9, use_bias=True):
    """测试单个种子的模型表现，返回最终准确率"""
    np.random.seed(seed)  # 设置当前种子
    
    # 生成数据并转换为批格式
    X, y = generate_data(task)
    X_batch = X.T  # (input_dim, batch_size) → (2,4)
    y_batch = y.T  # (output_dim, batch_size) → (1,4)
    input_dim, output_dim = X.shape[1], y.shape[1]
    
    # 初始化模型
    model = discPC_MultiLayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_bias=use_bias
    )
    
    # 训练模型
    for epoch in range(n_epochs):
        h, y_pred = model.initialize_activity(X_batch)
        h, y_pred = model.update_activity(
            X_batch, h, y_pred, target=y_batch,
            lr=lr_activity, m=m, lambda_target=lambda_target
        )
        model.update_weight(X_batch, h, y_pred, y_batch, lr=lr_weight)
    
    # 测试准确率
    pred_batch = model.predict(X_batch, m=m)  # 预测时使用与训练相同的m值
    correct = np.sum(pred_batch == y_batch)
    accuracy = correct / X.shape[0]  # X.shape[0]为样本数（4）
    return accuracy

if __name__ == "__main__":
    # 测试配置
    task = 'XOR'
    seed_range = range(100)  # 测试0-99共100个种子
    n_epochs = 5000
    hidden_dim = 2
    m = 50
    lr_activity = 0.1
    lr_weight = 0.1
    lambda_target = 0.9
    use_bias = True
    
    # 遍历种子测试
    successful_seeds = []
    print(f"开始测试{len(seed_range)}个随机种子（{task}任务）...\n")
    for seed in seed_range:
        acc = test_seed(
            seed=seed,
            task=task,
            n_epochs=n_epochs,
            hidden_dim=hidden_dim,
            m=m,
            lr_activity=lr_activity,
            lr_weight=lr_weight,
            lambda_target=lambda_target,
            use_bias=use_bias
        )
        if acc == 1.0:
            successful_seeds.append(seed)
        print(f"种子 {seed:3d} → 准确率: {acc:.4f} {'✓' if acc == 1.0 else ''}")
    
    # 输出统计结果
    print(f"\n===== 测试总结 =====")
    print(f"总测试种子数: {len(seed_range)}")
    print(f"成功种子数（准确率=1.0）: {len(successful_seeds)}")
    print(f"成功率: {len(successful_seeds)/len(seed_range):.2%}")
    print(f"成功的种子: {successful_seeds if successful_seeds else '无'}")