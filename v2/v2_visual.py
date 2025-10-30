import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
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
# 2. 多层discPC网络类（新增隐藏层）
# --------------------------
class discPC_MultiLayer:
    def __init__(self, input_dim, hidden_dim, output_dim, activation=sigmoid, use_bias=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        
        # 初始化权重（输入→隐藏、隐藏→输出）
        self.V1 = np.random.normal(0, np.sqrt(1 / self.input_dim), (self.input_dim, self.hidden_dim))
        self.V2 = np.random.normal(0, np.sqrt(1 / self.hidden_dim), (self.hidden_dim, self.output_dim))
        # 初始化偏置（隐藏层、输出层）
        self.b1 = np.zeros((hidden_dim, 1)) if use_bias else None  # 隐藏层偏置
        self.b2 = np.zeros((output_dim, 1)) if use_bias else None  # 输出层偏置
    
    def initialize_activity(self, x1):
        # 隐藏层活动值计算
        h_input = self.V1.T @ x1
        if self.use_bias:
            h_input += self.b1
        h = self.activation(h_input)
        
        # 输出层活动值计算
        y_input = self.V2.T @ h
        if self.use_bias:
            y_input += self.b2
        y = self.activation(y_input)
        
        return h, y  # 返回隐藏层和输出层活动值
    
    def update_activity(self, x1, h, y, target=None, lr=0.1, m=20, lambda_target=0.9):
        for _ in range(m):
            # 重新计算当前预测
            h_input = self.V1.T @ x1
            if self.use_bias:
                h_input += self.b1
            h_pred = self.activation(h_input)
            
            y_input = self.V2.T @ h_pred
            if self.use_bias:
                y_input += self.b2
            y_pred = self.activation(y_input)
            
            # 更新输出层活动值
            epsilon_y = y - y_pred
            update_y = -lr * epsilon_y
            if target is not None:
                target_error_y = y - target
                update_y -= lr * lambda_target * target_error_y
            y = y + update_y
            
            # 更新隐藏层活动值（反向传播输出层误差）
            epsilon_h = h - h_pred
            update_h = -lr * epsilon_h
            if target is not None:
                # 隐藏层误差 = 输出层误差×V2^T × sigmoid导数
                sigmoid_deriv_h = h_pred * (1 - h_pred)
                target_error_h = (self.V2 @ target_error_y) * sigmoid_deriv_h
                update_h -= lr * lambda_target * target_error_h
            h = h + update_h
        
        return h, y
    
    def update_weight(self, x1, h, y, target, lr=0.01):
        # 计算各层预测值
        h_input = self.V1.T @ x1
        if self.use_bias:
            h_input += self.b1
        h_pred = self.activation(h_input)
        
        y_input = self.V2.T @ h_pred
        if self.use_bias:
            y_input += self.b2
        y_pred = self.activation(y_input)
        
        # 计算各层误差（反向传播）
        error_y = target - y_pred  # 输出层误差
        sigmoid_deriv_h = h_pred * (1 - h_pred)
        error_h = (self.V2 @ error_y) * sigmoid_deriv_h  # 隐藏层误差
        
        # 更新权重和偏置
        self.V2 += lr * (h_pred @ error_y.T)  # 隐藏→输出层权重
        self.V1 += lr * (x1 @ error_h.T)      # 输入→隐藏层权重
        
        if self.use_bias:
            self.b2 += lr * error_y  # 输出层偏置
            self.b1 += lr * error_h  # 隐藏层偏置
    
    def predict(self, x1, m=20):
        h, y = self.initialize_activity(x1)
        h, y = self.update_activity(x1, h, y, m=m)
        return (y > 0.5).astype(float)

# --------------------------
# 3. 训练、测试与可视化（适配多层网络）
# --------------------------
if __name__ == "__main__":
    # 超参数配置（新增隐藏层维度）
    task = 'XOR'  # 重点测试XOR，可切换其他任务
    n_epochs = 5000  # 多层网络需更多训练轮次
    hidden_dim = 2  # XOR问题2个隐藏单元足够
    m = 8
    lr_activity = 0.1
    lr_weight = 0.1
    lambda_target = 0.9
    use_bias = True  # 偏置开关
    
    # 生成数据
    X, y = generate_data(task)
    input_dim, output_dim = X.shape[1], y.shape[1]
    
    # 初始化多层网络
    model = discPC_MultiLayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_bias=use_bias
    )
    
    # 初始化历史记录（适配多层参数）
    weights1_history = []  # 输入→隐藏层权重历史
    weights2_history = []  # 隐藏→输出层权重历史
    bias1_history = []     # 隐藏层偏置历史
    bias2_history = []     # 输出层偏置历史
    loss_history = []
    
    # 训练循环
    print(f"开始训练{task}任务（多层网络，偏置项{'启用' if use_bias else '禁用'}）...")
    for epoch in range(n_epochs):
        total_loss = 0
        correct_predictions = 0
        
        for i in range(len(X)):
            x1 = X[i].reshape(-1, 1)
            target = y[i].reshape(-1, 1)
            
            # 初始化并更新活动值
            h, y_pred = model.initialize_activity(x1)
            h, y_pred = model.update_activity(
                x1, h, y_pred, target=target,
                lr=lr_activity, m=m, lambda_target=lambda_target
            )
            
            # 更新权重
            model.update_weight(x1, h, y_pred, target, lr=lr_weight)
            
            # 计算损失和准确率
            loss = np.mean((y_pred - target) ** 2)
            total_loss += loss
            pred_label = (y_pred > 0.5).astype(float)
            if np.array_equal(pred_label, target):
                correct_predictions += 1
        
        # 记录历史数据（禁用偏置时填充0）
        weights1_history.append(model.V1.copy())
        weights2_history.append(model.V2.copy())
        bias1_history.append(model.b1.copy() if use_bias else np.zeros((hidden_dim, 1)))
        bias2_history.append(model.b2.copy() if use_bias else np.zeros((output_dim, 1)))
        loss_history.append(total_loss / len(X))
        
        # 打印训练进度（每500轮输出一次）
        if (epoch + 1) % 500 == 0:
            avg_loss = total_loss / len(X)
            accuracy = correct_predictions / len(X)
            print(f"Epoch {epoch+1:4d}, 平均损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
    
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
    
    # --------------------------
    # 4. 可视化（适配多层网络参数）
    # --------------------------
    # 提取历史数据（简化为单值可视化，隐藏层多单元取第一个）
    # 输入→隐藏层权重（2个输入×2个隐藏单元，取w11和w21）
    w11_history = [w[0, 0] for w in weights1_history]
    w21_history = [w[1, 0] for w in weights1_history]
    # 隐藏→输出层权重（2个隐藏单元×1个输出，取w1和w2）
    w_h1o_history = [w[0, 0] for w in weights2_history]
    w_h2o_history = [w[1, 0] for w in weights2_history]
    # 偏置（隐藏层取第一个，输出层取唯一值）
    b1_history = [b[0, 0] for b in bias1_history]
    b2_history = [b[0, 0] for b in bias2_history]
    
    # 创建画布（4个子图，展示多层参数变化）
    plt.figure(figsize=(12, 18))
    
    # 子图1：输入→隐藏层权重变化
    plt.subplot(4, 1, 1)
    plt.plot(range(n_epochs), w11_history, label='w11 (输入1→隐藏1)', color='blue')
    plt.plot(range(n_epochs), w21_history, label='w21 (输入2→隐藏1)', color='orange')
    plt.xlabel('训练轮次 (Epoch)')
    plt.ylabel('权重值')
    plt.title(f'{task}任务：输入→隐藏层权重变化')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 子图2：隐藏→输出层权重变化
    plt.subplot(4, 1, 2)
    plt.plot(range(n_epochs), w_h1o_history, label='w_h1o (隐藏1→输出)', color='green')
    plt.plot(range(n_epochs), w_h2o_history, label='w_h2o (隐藏2→输出)', color='red')
    plt.xlabel('训练轮次 (Epoch)')
    plt.ylabel('权重值')
    plt.title(f'{task}任务：隐藏→输出层权重变化')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 子图3：偏置项变化（适配偏置开关）
    plt.subplot(4, 1, 3)
    if use_bias:
        plt.plot(range(n_epochs), b1_history, label='b1 (隐藏层偏置)', color='purple')
        plt.plot(range(n_epochs), b2_history, label='b2 (输出层偏置)', color='brown', linestyle='--')
    plt.plot(range(n_epochs), loss_history, label='平均损失', color='red', linestyle=':')
    plt.xlabel('训练轮次 (Epoch)')
    plt.ylabel('数值')
    plt.title(f'{task}任务（偏置{"启用" if use_bias else "禁用"}）：偏置与损失变化')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 子图4：输入→隐藏层权重空间轨迹
    plt.subplot(4, 1, 4)
    cmap = LinearSegmentedColormap.from_list("epoch_cmap", ["blue", "purple", "red"])
    colors = cmap(np.linspace(0, 1, n_epochs))
    
    plt.scatter(w11_history, w21_history, c=colors, s=5, alpha=0.7, label='训练轨迹')
    plt.plot(w11_history, w21_history, color='gray', linewidth=0.5, alpha=0.5)
    plt.scatter(w11_history[0], w21_history[0], color='blue', s=50, marker='o', label='初始权重')
    plt.scatter(w11_history[-1], w21_history[-1], color='red', s=50, marker='*', label='收敛权重')
    
    plt.xlabel('w11 (输入1→隐藏1)')
    plt.ylabel('w21 (输入2→隐藏1)')
    plt.title(f'{task}任务：输入→隐藏层权重空间轨迹')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()