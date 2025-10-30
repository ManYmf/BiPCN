import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 1. 定义激活函数及数据集
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
# 2. 多层discPC网络类
# --------------------------
class discPC_MultiLayer:
    def __init__(self, input_dim, hidden_dim, output_dim, activation=sigmoid, use_bias=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        
        # 初始化权重
        self.V1 = np.random.normal(0, np.sqrt(1 / self.input_dim), (self.input_dim, self.hidden_dim))
        self.V2 = np.random.normal(0, np.sqrt(1 / self.hidden_dim), (self.hidden_dim, self.output_dim))
       
        # 初始化偏置
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
# 3. 种子遍历测试功能
# --------------------------
def test_random_seeds(num_seeds=50, task='XOR', n_epochs=3000, hidden_dim=2, 
                      m=8, lr_activity=0.1, lr_weight=0.1, lambda_target=0.9, use_bias=True):
    """测试多个随机种子下模型的性能"""
    X, y = generate_data(task)
    input_dim, output_dim = X.shape[1], y.shape[1]
    
    # 存储每个种子的结果
    results = {
        'seeds': [],
        'accuracies': [],
        'final_weights': [],
        'success': []  # 是否达到100%准确率
    }
    
    print(f"开始测试{num_seeds}个随机种子的{task}任务表现...")
    for seed in range(num_seeds):
        # 设置随机种子
        np.random.seed(seed)
        
        # 初始化模型
        model = discPC_MultiLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_bias=use_bias
        )
        
        # 训练模型
        for epoch in range(n_epochs):
            for i in range(len(X)):
                x1 = X[i].reshape(-1, 1)
                target = y[i].reshape(-1, 1)
                
                h, y_pred = model.initialize_activity(x1)
                h, y_pred = model.update_activity(
                    x1, h, y_pred, target=target,
                    lr=lr_activity, m=m, lambda_target=lambda_target
                )
                model.update_weight(x1, h, y_pred, target, lr=lr_weight)
        
        # 评估模型
        correct_count = 0
        for i in range(len(X)):
            x1 = X[i].reshape(-1, 1)
            pred = model.predict(x1, m=m)
            if np.array_equal(pred, y[i].reshape(-1, 1)):
                correct_count += 1
        
        accuracy = correct_count / len(X)
        
        # 保存结果
        results['seeds'].append(seed)
        results['accuracies'].append(accuracy)
        results['final_weights'].append((model.V1.copy(), model.V2.copy()))
        results['success'].append(accuracy == 1.0)
        
        # 打印进度
        if (seed + 1) % 10 == 0:
            print(f"已测试{seed + 1}/{num_seeds}个种子，其中{sum(results['success'])}个达到100%准确率")
    
    return results

# --------------------------
# 4. 结果可视化
# --------------------------
def visualize_seed_results(results):
    """可视化种子测试结果"""
    plt.figure(figsize=(15, 10))
    
    # 子图1：各种子准确率
    plt.subplot(2, 2, 1)
    colors = ['green' if acc == 1.0 else 'red' for acc in results['accuracies']]
    plt.scatter(results['seeds'], results['accuracies'], c=colors, alpha=0.7)
    plt.axhline(y=1.0, color='green', linestyle='--', label='100%准确率')
    plt.axhline(y=0.75, color='orange', linestyle='--', label='75%准确率')
    plt.xlabel('随机种子')
    plt.ylabel('最终准确率')
    plt.title('不同随机种子的模型准确率')
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
    plt.title('准确率分布统计')
    plt.grid(alpha=0.3, axis='y')
    
    # 子图3：成功与失败种子的比例
    plt.subplot(2, 2, 3)
    success_rate = sum(results['success']) / len(results['success'])
    labels = ['成功 (100%)', '失败 (<100%)']
    sizes = [success_rate, 1 - success_rate]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['green', 'red'])
    plt.title('种子成功率分布')
    
    # 子图4：成功种子与失败种子的初始权重对比
    plt.subplot(2, 2, 4)
    # 提取成功和失败种子的第一个权重参数
    success_weights = []
    fail_weights = []
    
    for i, success in enumerate(results['success']):
        # 取输入到隐藏层的第一个权重作为代表
        w = results['final_weights'][i][0][0, 0]  # V1[0,0]
        if success:
            success_weights.append(w)
        else:
            fail_weights.append(w)
    
    plt.hist(success_weights, bins=10, alpha=0.5, label='成功种子', color='green')
    plt.hist(fail_weights, bins=10, alpha=0.5, label='失败种子', color='red')
    plt.xlabel('输入→隐藏层权重 (V1[0,0])')
    plt.ylabel('频率')
    plt.title('成功与失败种子的权重分布')
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print("\n===== 种子测试统计结果 =====")
    print(f"测试种子总数: {len(results['seeds'])}")
    print(f"达到100%准确率的种子数: {sum(results['success'])}")
    print(f"成功率: {sum(results['success'])/len(results['seeds']):.2%}")
    print(f"主要失败准确率: {[f'{acc:.0%}' for acc in set(results['accuracies']) if acc < 1.0]}")
    
    # 输出成功的种子列表
    success_seeds = [results['seeds'][i] for i, success in enumerate(results['success']) if success]
    print(f"\n成功的种子列表: {success_seeds[:10]}...")  # 只显示前10个

# --------------------------
# 5. 主程序：运行种子测试
# --------------------------
if __name__ == "__main__":
    # 配置测试参数
    num_seeds = 100  # 测试的种子数量
    task = 'XOR'     # 测试任务
    n_epochs = 3000  # 每个种子的训练轮次
    
    # 运行种子测试
    results = test_random_seeds(
        num_seeds=num_seeds,
        task=task,
        n_epochs=n_epochs,
        hidden_dim=2,
        m=8,
        lr_activity=0.1,
        lr_weight=0.1,
        lambda_target=0.9,
        use_bias=True
    )
    
    # 可视化结果
    visualize_seed_results(results)