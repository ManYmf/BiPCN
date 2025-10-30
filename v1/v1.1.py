import numpy as np

# --------------------------
# 1. 定义激活函数及数据集
# --------------------------
def sigmoid(x):
    """Sigmoid激活函数，范围在[0,1]，更适合二分类问题"""
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))  # 避免数值溢出

def generate_data(task='AND'):
    """生成AND/OR/XOR问题的输入和标签"""
    X = np.array([[0, 0],  # 输入特征（2维）
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=np.float32)
    if task == 'AND':
        y = np.array([[0], [0], [0], [1]], dtype=np.float32)  # AND标签
    elif task == 'OR':
        y = np.array([[0], [1], [1], [1]], dtype=np.float32)  # OR标签
    elif task == 'XOR':
        y = np.array([[0], [1], [1], [0]], dtype=np.float32)  # XOR标签
    else:
        raise ValueError("任务必须是'AND'、'OR'或'XOR'")

    return X, y

# --------------------------
# 2. discPC网络类（带偏置项开关）
# --------------------------
class discPC:
    def __init__(self, input_dim, output_dim, activation=sigmoid, use_bias=True):
        self.input_dim = input_dim  # 输入层维度（2）
        self.output_dim = output_dim  # 输出层维度（1）
        self.activation = activation
        self.use_bias = use_bias  # 偏置项开关
        
        # 初始化自下而上权重V1（输入层→输出层，使用较小的随机值）
        self.V1 = np.random.normal(0, 0.1, (input_dim, output_dim))
        
        # 根据开关初始化偏置项（启用时初始化为0，禁用时设为None）
        self.b1 = np.zeros((output_dim, 1)) if use_bias else None
    
    def initialize_activity(self, x1):
        """前馈扫描初始化高层活动（输出层x2）"""
        # 计算基础输入：V1.T @ x1
        base_input = self.V1.T @ x1
        # 若启用偏置则加上偏置项
        if self.use_bias:
            base_input += self.b1
        # 应用激活函数
        x2_initial = self.activation(base_input)
        return x2_initial
    
    def update_activity(self, x1, x2, target=None, lr=0.1, m=20, lambda_target=0.9):
        """神经活动梯度下降更新（m次迭代）"""
        for _ in range(m):
            # 计算自下而上预测值：f(V1 @ x1 + b1)（根据偏置开关决定是否加b1）
            base_pred = self.V1.T @ x1
            if self.use_bias:
                base_pred += self.b1
            pred = self.activation(base_pred)
            
            # 预测误差
            epsilon = x2 - pred
            
            # 基础更新项：最小化自下而上预测误差
            update = -lr * epsilon
            
            # 如果提供了目标标签，添加目标约束项
            if target is not None:
                target_error = x2 - target
                update -= lr * lambda_target * target_error
        
            # 更新神经活动
            x2 = x2 + update
        return x2
    
    def update_weight(self, x1, x2, target, lr=0.01):
        """Hebbian规则更新权重V1和偏置b1（根据开关决定是否更新偏置）"""
        # 计算预测值（根据偏置开关决定是否加b1）
        base_pred = self.V1.T @ x1
        if self.use_bias:
            base_pred += self.b1
        pred = self.activation(base_pred)
        
        # 后突触误差：目标与预测之间的误差
        error = target - pred
        
        # 前突触活动：输入层活动
        pre_synaptic = x1
        
        # 更新权重V1
        self.V1 += lr * (pre_synaptic @ error.T)
        
        # 若启用偏置，更新偏置项
        if self.use_bias:
            self.b1 += lr * error
    
    def predict(self, x1, m=20):
        """测试阶段：初始化后更新m次活动，输出预测结果"""
        x2 = self.initialize_activity(x1)  # 初始化
        x2 = self.update_activity(x1, x2, m=m)  # 更新m次
        return (x2 > 0.5).astype(float)  # 阈值判断（>0.5为1）

# --------------------------
# 3. 训练与测试（可通过use_bias参数控制偏置）
# --------------------------
if __name__ == "__main__":
    # 超参数
    task = 'OR'  # 可切换为'AND'/'XOR'
    n_epochs = 2000  
    m = 50  
    lr_activity = 0.1  
    lr_weight = 0.1  
    lambda_target = 0.9  
    use_bias = False  # 偏置开关（True启用，False禁用）
    
    # 生成数据
    X, y = generate_data(task)
    input_dim, output_dim = X.shape[1], y.shape[1]
    
    # 初始化网络（传入偏置开关参数）
    model = discPC(input_dim, output_dim, use_bias=use_bias)
    
    # 训练循环
    print(f"开始训练（偏置项{'启用' if use_bias else '禁用'}）...")
    for epoch in range(n_epochs):
        total_loss = 0
        correct_predictions = 0
        
        for i in range(len(X)):
            x1 = X[i].reshape(-1, 1)  # 输入层活动（固定）
            target = y[i].reshape(-1, 1)  # 目标输出
            
            # 步骤1：初始化输出层活动
            x2 = model.initialize_activity(x1)
            
            # 步骤2：更新神经活动m次（包含目标约束）
            x2 = model.update_activity(x1, x2, target=target, lr=lr_activity, m=m, lambda_target=lambda_target)
            
            # 步骤3：更新权重（根据开关决定是否更新偏置）
            model.update_weight(x1, x2, target, lr=lr_weight)
            
            # 计算损失和预测准确率
            loss = np.mean((x2 - target) **2)
            total_loss += loss
            
            # 检查预测是否正确
            prediction = (x2 > 0.5).astype(float)
            if np.array_equal(prediction, target):
                correct_predictions += 1
        
        # 每200轮打印一次训练进度
        if (epoch + 1) % 200 == 0:
            avg_loss = total_loss / len(X)
            accuracy = correct_predictions / len(X)
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # 测试：对每个样本更新m次活动后预测
    print(f"\n{task}问题测试结果（偏置项{'启用' if use_bias else '禁用'}）：")
    correct_count = 0
    for i in range(len(X)):
        x1 = X[i].reshape(-1, 1)
        pred = model.predict(x1, m=m)
        is_correct = np.array_equal(pred, y[i].reshape(-1, 1))
        if is_correct:
            correct_count += 1
        print(f"输入: {X[i]}, 预测: {pred[0][0]}, 真实: {y[i][0]}, {'✓' if is_correct else '✗'}")
    
    # 打印最终准确率
    final_accuracy = correct_count / len(X)
    print(f"\n最终测试准确率: {final_accuracy:.4f}")