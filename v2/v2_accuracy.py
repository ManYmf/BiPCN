import numpy as np

# --------------------------
# 1. 定义激活函数及数据集（保持不变）
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
# 2. 多层discPC网络类（保持不变）
# --------------------------
class discPC_MultiLayer:
    def __init__(self, input_dim, hidden_dim, output_dim, activation=sigmoid, use_bias=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        
        self.V1 = np.random.normal(0, 0.1, (input_dim, hidden_dim))
        self.V2 = np.random.normal(0, 0.1, (hidden_dim, output_dim))
        
        self.b1 = np.zeros((hidden_dim, 1)) if use_bias else None
        self.b2 = np.zeros((output_dim, 1)) if use_bias else None
    
    def initialize_activity(self, x1):
        h_input = self.V1.T @ x1
        if self.use_bias:
            h_input += self.b1
        h = self.activation(h_input)
        
        y_input = self.V2.T @ h
        if self.use_bias:
            y_input += self.b2
        y = self.activation(y_input)
        
        return h, y
    
    def update_activity(self, x1, h, y, target=None, lr=0.1, m=20, lambda_target=0.9):
        for _ in range(m):
            h_input = self.V1.T @ x1
            if self.use_bias:
                h_input += self.b1
            h_pred = self.activation(h_input)
            
            y_input = self.V2.T @ h_pred
            if self.use_bias:
                y_input += self.b2
            y_pred = self.activation(y_input)
            
            epsilon_y = y - y_pred
            update_y = -lr * epsilon_y
            if target is not None:
                target_error_y = y - target
                update_y -= lr * lambda_target * target_error_y
            y = y + update_y
            
            epsilon_h = h - h_pred
            update_h = -lr * epsilon_h
            if target is not None:
                sigmoid_deriv_h = h_pred * (1 - h_pred)
                target_error_h = (self.V2 @ target_error_y) * sigmoid_deriv_h
                update_h -= lr * lambda_target * target_error_h
            h = h + update_h
        
        return h, y
    
    def update_weight(self, x1, h, y, target, lr=0.01):
        h_input = self.V1.T @ x1
        if self.use_bias:
            h_input += self.b1
        h_pred = self.activation(h_input)
        
        y_input = self.V2.T @ h_pred
        if self.use_bias:
            y_input += self.b2
        y_pred = self.activation(y_input)
        
        error_y = target - y_pred
        sigmoid_deriv_h = h_pred * (1 - h_pred)
        error_h = (self.V2 @ error_y) * sigmoid_deriv_h
        
        self.V2 += lr * (h_pred @ error_y.T)
        self.V1 += lr * (x1 @ error_h.T)
        
        if self.use_bias:
            self.b2 += lr * error_y
            self.b1 += lr * error_h
    
    def predict(self, x1, m=20):
        h, y = self.initialize_activity(x1)
        h, y = self.update_activity(x1, h, y, m=m)  # 无target，纯靠网络自身调整
        return (y > 0.5).astype(float)

# --------------------------
# 3. 训练与测试（修改准确率统计逻辑）
# --------------------------
if __name__ == "__main__":
    # 超参数配置
    task = 'XOR'
    n_epochs = 5000  # 可对比500和5000轮的差异
    hidden_dim = 2
    m = 50
    lr_activity = 0.1
    lr_weight = 0.1
    lambda_target = 0.9
    use_bias = True
    
    # 生成数据
    X, y = generate_data(task)
    input_dim, output_dim = X.shape[1], y.shape[1]
    
    # 初始化模型
    model = discPC_MultiLayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_bias=use_bias
    )
    
    # 仅保留损失历史记录（用于打印）
    loss_history = []
    
    # 训练循环
    print(f"开始训练{task}任务（多层网络，偏置项{'启用' if use_bias else '禁用'}）...")
    for epoch in range(n_epochs):
        total_loss = 0
        correct_predictions = 0  # 用于统计训练时的真实预测准确率
        
        for i in range(len(X)):
            x1 = X[i].reshape(-1, 1)
            target = y[i].reshape(-1, 1)
            
            # 1. 初始化并更新活动值（带target，用于权重更新）
            h, y_pred_adjusted = model.initialize_activity(x1)
            h, y_pred_adjusted = model.update_activity(
                x1, h, y_pred_adjusted, target=target,
                lr=lr_activity, m=m, lambda_target=lambda_target
            )
            
            # 2. 更新权重（基于带target调整的活动值）
            model.update_weight(x1, h, y_pred_adjusted, target, lr=lr_weight)
            
            # 3. 计算损失（仍用调整后的活动值，因为损失用于指导权重更新）
            loss = np.mean((y_pred_adjusted - target) ** 2)
            total_loss += loss
            
            # 4. 关键改动：训练时的准确率统计——用模型独立预测（无target）的结果
            pred_label = model.predict(x1, m=m)  # 和测试时逻辑一致
            if np.array_equal(pred_label, target):
                correct_predictions += 1
        
        # 记录平均损失
        avg_loss = total_loss / len(X)
        loss_history.append(avg_loss)
        
        # 打印训练进度（每100轮输出一次）
        if (epoch + 1) % 100 == 0:
            accuracy = correct_predictions / len(X)
            print(f"Epoch {epoch+1:4d}, 平均损失: {avg_loss:.4f}, 训练准确率: {accuracy:.4f}")
    
    # 测试结果
    print(f"\n{task}任务测试结果（多层网络，偏置项{'启用' if use_bias else '禁用'}）：")
    correct_count = 0
    for i in range(len(X)):
        x1 = X[i].reshape(-1, 1)
        pred = model.predict(x1, m=m)
        is_correct = np.array_equal(pred, y[i].reshape(-1, 1))
        correct_count += 1 if is_correct else 0
        print(f"输入: {X[i]}, 预测: {pred[0][0]}, 真实: {y[i][0]}, {'✓' if is_correct else '✗'}")
    print(f"\n最终测试准确率: {correct_count/len(X):.4f}")