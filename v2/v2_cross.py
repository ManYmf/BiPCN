import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 1. 定义激活函数及数据集
# --------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))  # 防止指数溢出

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
# 2. 改用交叉熵损失的多层discPC网络类
# --------------------------
class discPC_MultiLayer_CrossEntropy:
    def __init__(self, input_dim, hidden_dim, output_dim, activation=sigmoid, use_bias=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        
        # 采用Xavier初始化（更适合Sigmoid+交叉熵）
        self.V1 = np.random.normal(0, np.sqrt(1 / input_dim), (input_dim, hidden_dim))  # 输入→隐藏
        self.V2 = np.random.normal(0, np.sqrt(1 / hidden_dim), (hidden_dim, output_dim))  # 隐藏→输出
        
        # 初始化偏置
        self.b1 = np.zeros((hidden_dim, 1)) if use_bias else None
        self.b2 = np.zeros((output_dim, 1)) if use_bias else None
    
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
        
        return h, y
    
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
            
            # 更新隐藏层活动值
            epsilon_h = h - h_pred
            update_h = -lr * epsilon_h
            if target is not None:
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
        
        # --------------------------
        # 核心修改：交叉熵损失的误差计算
        # 对Sigmoid输出，交叉熵损失的导数 = y_pred - target（简化形式）
        # --------------------------
        error_y = y_pred - target  # 交叉熵误差（与MSE的target - y_pred符号相反）
        
        # 隐藏层误差仍需结合激活函数导数
        sigmoid_deriv_h = h_pred * (1 - h_pred)
        error_h = (self.V2 @ error_y) * sigmoid_deriv_h
        
        # 更新权重（注意符号：此处已包含损失函数的导数方向）
        self.V2 -= lr * (h_pred @ error_y.T)  # 隐藏→输出层权重（减去梯度）
        self.V1 -= lr * (x1 @ error_h.T)      # 输入→隐藏层权重（减去梯度）
        
        if self.use_bias:
            self.b2 -= lr * error_y  # 输出层偏置
            self.b1 -= lr * error_h  # 隐藏层偏置
    
    def predict(self, x1, m=20):
        h, y = self.initialize_activity(x1)
        h, y = self.update_activity(x1, h, y, m=m)
        return (y > 0.5).astype(float)
    
    @staticmethod
    def cross_entropy_loss(y_pred, target, epsilon=1e-10):
        # 计算交叉熵损失（添加epsilon避免log(0)）
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(target * np.log(y_pred) + (1 - target) * np.log(1 - y_pred))

# --------------------------
# 3. 种子遍历测试功能（适配交叉熵）
# --------------------------
def test_random_seeds(num_seeds=50, task='XOR', n_epochs=3000, hidden_dim=2, 
                      m=8, lr_activity=0.1, lr_weight=0.1, lambda_target=0.9, use_bias=True):
    X, y = generate_data(task)
    input_dim, output_dim = X.shape[1], y.shape[1]
    
    results = {
        'seeds': [],
        'accuracies': [],
        'final_weights': [],
        'success': [],
        'loss_history': []  # 新增：记录每个种子的最终损失
    }
    
    print(f"开始测试{num_seeds}个随机种子的{task}任务（交叉熵损失）...")
    for seed in range(num_seeds):
        np.random.seed(seed)
        model = discPC_MultiLayer_CrossEntropy(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_bias=use_bias
        )
        
        # 训练模型
        epoch_losses = []
        for epoch in range(n_epochs):
            total_loss = 0
            for i in range(len(X)):
                x1 = X[i].reshape(-1, 1)
                target = y[i].reshape(-1, 1)
                
                h, y_pred = model.initialize_activity(x1)
                h, y_pred = model.update_activity(
                    x1, h, y_pred, target=target,
                    lr=lr_activity, m=m, lambda_target=lambda_target
                )
                model.update_weight(x1, h, y_pred, target, lr=lr_weight)
                
                # 记录交叉熵损失
                total_loss += model.cross_entropy_loss(y_pred, target)
            epoch_losses.append(total_loss / len(X))
        
        # 评估模型
        correct_count = 0
        for i in range(len(X)):
            x1 = X[i].reshape(-1, 1)
            pred = model.predict(x1, m=m)
            if np.array_equal(pred, y[i].reshape(-1, 1)):
                correct_count += 1
        
        accuracy = correct_count / len(X)
        results['seeds'].append(seed)
        results['accuracies'].append(accuracy)
        results['final_weights'].append((model.V1.copy(), model.V2.copy()))
        results['success'].append(accuracy == 1.0)
        results['loss_history'].append(epoch_losses)
        
        if (seed + 1) % 10 == 0:
            print(f"已测试{seed + 1}/{num_seeds}个种子，其中{sum(results['success'])}个达到100%准确率")
    
    return results

# --------------------------
# 4. 结果可视化（适配交叉熵）
# --------------------------
def visualize_seed_results(results):
    plt.figure(figsize=(15, 12))
    
    # 子图1：各种子准确率
    plt.subplot(2, 2, 1)
    colors = ['green' if acc == 1.0 else 'red' for acc in results['accuracies']]
    plt.scatter(results['seeds'], results['accuracies'], c=colors, alpha=0.7)
    plt.axhline(y=1.0, color='green', linestyle='--', label='100%准确率')
    plt.axhline(y=0.75, color='orange', linestyle='--', label='75%准确率')
    plt.xlabel('随机种子')
    plt.ylabel('最终准确率')
    plt.title('不同随机种子的模型准确率（交叉熵损失）')
    plt.ylim(0.7, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 子图2：准确率分布
    plt.subplot(2, 2, 2)
    acc_counts = {}
    for acc in results['accuracies']:
        key = f"{acc:.0%}"
        acc_counts[key] = acc_counts.get(key, 0) + 1
    plt.bar(acc_counts.keys(), acc_counts.values(), color=['red', 'green'] if 0.75 in results['accuracies'] else ['green'])
    plt.xlabel('准确率')
    plt.ylabel('种子数量')
    plt.title('准确率分布统计（交叉熵损失）')
    plt.grid(alpha=0.3, axis='y')
    
    # 子图3：成功与失败种子的比例
    plt.subplot(2, 2, 3)
    success_rate = sum(results['success']) / len(results['success'])
    labels = ['成功 (100%)', '失败 (<100%)']
    sizes = [success_rate, 1 - success_rate]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['green', 'red'])
    plt.title('种子成功率分布（交叉熵损失）')
    
    # 子图4：成功与失败种子的损失曲线对比
    plt.subplot(2, 2, 4)
    # 取前5个成功和失败的种子损失曲线
    success_indices = [i for i, s in enumerate(results['success']) if s][:5]
    fail_indices = [i for i, s in enumerate(results['success']) if not s][:5]
    
    for i in success_indices:
        plt.plot(results['loss_history'][i], color='green', alpha=0.5, label='成功种子' if i == success_indices[0] else "")
    for i in fail_indices:
        plt.plot(results['loss_history'][i], color='red', alpha=0.5, label='失败种子' if i == fail_indices[0] else "")
    
    plt.xlabel('训练轮次')
    plt.ylabel('交叉熵损失')
    plt.title('成功与失败种子的损失曲线对比')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print("\n===== 交叉熵损失种子测试统计结果 =====")
    print(f"测试种子总数: {len(results['seeds'])}")
    print(f"达到100%准确率的种子数: {sum(results['success'])}")
    print(f"成功率: {sum(results['success'])/len(results['seeds']):.2%}")
    print(f"主要失败准确率: {[f'{acc:.0%}' for acc in set(results['accuracies']) if acc < 1.0]}")
    
    success_seeds = [results['seeds'][i] for i, success in enumerate(results['success']) if success]
    print(f"\n成功的种子列表: {success_seeds[:10]}...")

# --------------------------
# 5. 主程序：运行交叉熵损失测试
# --------------------------
if __name__ == "__main__":
    num_seeds = 100  # 测试的种子数量
    task = 'XOR'     # 测试任务
    n_epochs = 3000  # 训练轮次
    
    # 运行种子测试（交叉熵损失）
    results = test_random_seeds(
        num_seeds=num_seeds,
        task=task,
        n_epochs=n_epochs,
        hidden_dim=2,
        m=8,
        lr_activity=0.1,
        lr_weight=0.05,  # 交叉熵通常需要稍小的学习率
        lambda_target=0.9,
        use_bias=True
    )
    
    # 可视化结果
    visualize_seed_results(results)