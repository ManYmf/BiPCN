import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 1. 定义激活函数及数据集（不变）
# --------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def generate_data(task='AND'):
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
# 2. discPC网络类（添加偏置项开关）
# --------------------------
class discPC:
    def __init__(self, input_dim, output_dim, activation=sigmoid, use_bias=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias  # 偏置项开关（默认启用）
        
        # 初始化权重
        self.V1 = np.random.normal(0, 0.1, (input_dim, output_dim))
        # 根据开关初始化偏置（禁用时设为None）
        self.b1 = np.zeros((output_dim, 1)) if use_bias else None
    
    def initialize_activity(self, x1):
        base_input = self.V1.T @ x1
        if self.use_bias:
            base_input += self.b1
        return self.activation(base_input)
    
    def update_activity(self, x1, x2, target=None, lr=0.1, m=20, lambda_target=0.9):
        for _ in range(m):
            base_pred = self.V1.T @ x1
            if self.use_bias:
                base_pred += self.b1
            pred = self.activation(base_pred)
            
            epsilon = x2 - pred
            update = -lr * epsilon
            if target is not None:
                target_error = x2 - target
                update -= lr * lambda_target * target_error
            
            x2 = x2 + update
        return x2
    
    def update_weight(self, x1, x2, target, lr=0.01):
        base_pred = self.V1.T @ x1
        if self.use_bias:
            base_pred += self.b1
        pred = self.activation(base_pred)
        
        error = target - pred
        pre_synaptic = x1
        
        # 更新权重
        self.V1 += lr * (pre_synaptic @ error.T)
        # 仅在启用偏置时更新偏置项
        if self.use_bias:
            self.b1 += lr * error
    
    def predict(self, x1, m=20):
        x2 = self.initialize_activity(x1)
        x2 = self.update_activity(x1, x2, m=m)
        return (x2 > 0.5).astype(float)

# --------------------------
# 3. 训练、测试与可视化（适配偏置开关）
# --------------------------
if __name__ == "__main__":
    # 超参数（新增偏置开关）
    task = 'NOR'  # 可切换为'AND'/'OR'/'XOR'/'NOR'
    n_epochs = 2000
    m = 50
    lr_activity = 0.1
    lr_weight = 0.1
    lambda_target = 0.9
    use_bias = True  # 偏置开关（True启用，False禁用）
    
    # 生成数据
    X, y = generate_data(task)
    input_dim, output_dim = X.shape[1], y.shape[1]
    
    # 初始化网络（传入偏置开关）
    model = discPC(input_dim, output_dim, use_bias=use_bias)
    
    # 初始化历史记录（适配偏置开关）
    weights_history = []
    biases_history = []  # 禁用时存储全0，保证长度一致
    loss_history = []
    
    # 训练循环
    print(f"开始训练（偏置项{'启用' if use_bias else '禁用'}）...")
    for epoch in range(n_epochs):
        total_loss = 0
        correct_predictions = 0
        
        for i in range(len(X)):
            x1 = X[i].reshape(-1, 1)
            target = y[i].reshape(-1, 1)
            
            x2 = model.initialize_activity(x1)
            x2 = model.update_activity(x1, x2, target=target, lr=lr_activity, m=m, lambda_target=lambda_target)
            model.update_weight(x1, x2, target, lr=lr_weight)
            
            loss = np.mean((x2 - target) **2)
            total_loss += loss
            prediction = (x2 > 0.5).astype(float)
            if np.array_equal(prediction, target):
                correct_predictions += 1
        
        # 记录历史数据（禁用偏置时偏置值记为0）
        weights_history.append(model.V1.copy())
        if use_bias:
            biases_history.append(model.b1.copy())
        else:
            biases_history.append(np.array([[0.0]]))  # 填充0以保持长度一致
        loss_history.append(total_loss / len(X))
        
        # 打印训练进度
        if (epoch + 1) % 200 == 0:
            avg_loss = total_loss / len(X)
            accuracy = correct_predictions / len(X)
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # 测试（添加偏置状态提示）
    print(f"\n{task}问题测试结果（偏置项{'启用' if use_bias else '禁用'}）：")
    correct_count = 0
    for i in range(len(X)):
        x1 = X[i].reshape(-1, 1)
        pred = model.predict(x1, m=m)
        is_correct = np.array_equal(pred, y[i].reshape(-1, 1))
        correct_count += 1 if is_correct else 0
        print(f"输入: {X[i]}, 预测: {pred[0][0]}, 真实: {y[i][0]}, {'✓' if is_correct else '✗'}")
    print(f"\n最终测试准确率: {correct_count/len(X):.4f}")
    
    # --------------------------
    # 4. 可视化（适配偏置开关，图表动态调整）
    # --------------------------
    # 提取历史数据
    w1_history = [w[0, 0] for w in weights_history]
    w2_history = [w[1, 0] for w in weights_history]
    b_history = [b[0, 0] for b in biases_history]
    
    # 创建画布（3个子图，适配偏置状态）
    plt.figure(figsize=(12, 15))
    
    # 子图1：权重变化（不受偏置开关影响）
    plt.subplot(3, 1, 1)
    plt.plot(range(n_epochs), w1_history, label='w1 (输入特征1的权重)', color='blue')
    plt.plot(range(n_epochs), w2_history, label='w2 (输入特征2的权重)', color='orange')
    plt.xlabel('训练轮次 (Epoch)')
    plt.ylabel('权重值')
    plt.title(f'{task}任务（偏置{"启用" if use_bias else "禁用"}）：权重随训练轮次的变化')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 子图2：偏置与损失（禁用偏置时隐藏偏置曲线）
    plt.subplot(3, 1, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    if use_bias:
        # 启用偏置：显示偏置+损失双曲线
        line1 = ax1.plot(range(n_epochs), b_history, label='b1 (偏置)', color='green')[0]
        ax1.set_ylabel('偏置值', color='green')
        ax1.tick_params(axis='y', labelcolor='green')
        line2 = ax2.plot(range(n_epochs), loss_history, label='平均损失', color='red', linestyle='--')[0]
        ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc='upper right')
    else:
        # 禁用偏置：仅显示损失曲线
        line2 = ax2.plot(range(n_epochs), loss_history, label='平均损失', color='red', linestyle='-')[0]
        ax1.set_ylabel('平均损失', color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.legend([line2], [line2.get_label()], loc='upper right')
    
    ax2.set_ylabel('平均损失', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.xlabel('训练轮次 (Epoch)')
    plt.title(f'{task}任务（偏置{"启用" if use_bias else "禁用"}）：偏置与损失的变化')
    plt.grid(alpha=0.3)
    
    # 子图3：权重空间轨迹（不受偏置开关影响）
    plt.subplot(3, 1, 3)
    cmap = LinearSegmentedColormap.from_list("epoch_cmap", ["blue", "purple", "red"])
    colors = cmap(np.linspace(0, 1, n_epochs))
    
    plt.scatter(w1_history, w2_history, c=colors, s=5, alpha=0.7, label='训练轨迹')
    plt.plot(w1_history, w2_history, color='gray', linewidth=0.5, alpha=0.5)
    plt.scatter(w1_history[0], w2_history[0], color='blue', s=50, marker='o', label='起点 (初始权重)')
    plt.scatter(w1_history[-1], w2_history[-1], color='red', s=50, marker='*', label='终点 (收敛权重)')
    
    plt.xlabel('w1 (输入特征1的权重)')
    plt.ylabel('w2 (输入特征2的权重)')
    plt.title(f'{task}任务（偏置{"启用" if use_bias else "禁用"}）：权重空间轨迹图')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()