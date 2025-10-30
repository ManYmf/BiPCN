import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import time
import pandas as pd
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

# --------------------------
# 1. 激活函数
# --------------------------
def sigmoid(x):
    return torch.sigmoid(torch.clamp(x, -20, 20))

def softmax(x):
    exp_x = torch.exp(torch.clamp(x, -20, 20))
    return exp_x / torch.sum(exp_x, dim=0, keepdim=True)

# --------------------------
# 2. discPC网络类（简化版用于基准测试）
# --------------------------
class discPC_Simple:
    def __init__(self, layers, device='cpu'):
        self.layers = layers
        self.n_layers = len(layers)
        self.device = device
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            input_dim = layers[i]
            output_dim = layers[i+1]
            
            # Xavier初始化
            weight = torch.randn(input_dim, output_dim, device=device) * np.sqrt(2.0 / (input_dim + output_dim))
            self.weights.append(weight)
            
            bias = torch.zeros(output_dim, 1, device=device)
            self.biases.append(bias)
    
    def initialize_activity(self, x_batch):
        activities = [x_batch]
        
        for i in range(self.n_layers - 1):
            prev_activity = activities[-1]
            layer_input = self.weights[i].T @ prev_activity + self.biases[i]
            
            if i == self.n_layers - 1:  # 输出层
                activity = softmax(layer_input)
            else:  # 隐藏层
                activity = sigmoid(layer_input)
            
            activities.append(activity)
        
        return activities
    
    def update_activity(self, x_batch, activities, target=None, lr=0.1, m=20, lambda_target=0.9):
        current_activities = [act.clone() for act in activities]
        
        for _ in range(m):
            # 前向计算预测值
            pred_activities = [x_batch]
            for i in range(self.n_layers - 1):
                prev_pred = pred_activities[-1]
                layer_input = self.weights[i].T @ prev_pred + self.biases[i]
                
                if i == self.n_layers - 1:
                    pred_activities.append(softmax(layer_input))
                else:
                    pred_activities.append(sigmoid(layer_input))
            
            # 反向更新活动值
            for i in reversed(range(1, self.n_layers)):
                epsilon = current_activities[i] - pred_activities[i]
                update = -lr * epsilon
                
                if target is not None and i == self.n_layers - 1:
                    target_error = current_activities[i] - target
                    update -= lr * lambda_target * target_error
                elif target is not None and i < self.n_layers - 1:
                    sigmoid_deriv = pred_activities[i] * (1 - pred_activities[i])
                    next_error = current_activities[i+1] - target if i+1 == self.n_layers -1 else current_activities[i+1] - pred_activities[i+1]
                    target_error = (self.weights[i] @ next_error) * sigmoid_deriv
                    update -= lr * lambda_target * target_error
                
                current_activities[i] += update
        
        return current_activities
    
    def update_weight(self, x_batch, activities, target, lr=0.01):
        pred_activities = [x_batch]
        for i in range(self.n_layers - 1):
            prev_pred = pred_activities[-1]
            layer_input = self.weights[i].T @ prev_pred + self.biases[i]
            
            if i == self.n_layers - 1:
                pred_activities.append(softmax(layer_input))
            else:
                pred_activities.append(sigmoid(layer_input))
        
        # 计算误差
        errors = [None] * (self.n_layers - 1)
        errors[-1] = target - pred_activities[-1]
        
        for i in reversed(range(self.n_layers - 2)):
            sigmoid_deriv = pred_activities[i+1] * (1 - pred_activities[i+1])
            errors[i] = (self.weights[i+1] @ errors[i+1]) * sigmoid_deriv
        
        # 更新权重和偏置
        batch_size = x_batch.shape[1]
        for i in range(self.n_layers - 1):
            self.weights[i] += lr * (pred_activities[i] @ errors[i].T) / batch_size
            self.biases[i] += lr * torch.mean(errors[i], dim=1, keepdim=True)
    
    def predict(self, x_batch, m=20):
        activities = self.initialize_activity(x_batch)
        activities = self.update_activity(x_batch, activities, m=m)
        return torch.argmax(activities[-1], dim=0)

# --------------------------
# 3. 数据加载函数
# --------------------------
def load_dataset_for_benchmark(dataset_name, test_size=0.2):
    """加载数据集用于基准测试"""
    if dataset_name == 'wine':
        data = load_wine()
        X = data.data
        y = data.target
        class_names = data.target_names
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        X = data.data
        y = data.target
        class_names = data.target_names
    elif dataset_name == 'digits':
        data = load_digits()
        X = data.data
        y = data.target
        class_names = [str(i) for i in range(10)]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, class_names, scaler

# --------------------------
# 4. discPC训练函数
# --------------------------
def train_discpc(X_train, X_test, y_train, y_test, layers, n_epochs=1000, 
                m=10, lr_activity=0.05, lr_weight=0.05, lambda_target=0.8, device='cpu'):
    """训练discPC模型"""
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device).T
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device).T
    
    # 独热编码标签
    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_encoded = encoder.transform(y_test.reshape(-1, 1))
    
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.float32, device=device).T
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.float32, device=device).T
    
    # 创建模型
    model = discPC_Simple(layers, device=device)
    
    # 训练
    start_time = time.time()
    
    for epoch in range(n_epochs):
        activities = model.initialize_activity(X_train_tensor)
        activities = model.update_activity(
            X_train_tensor, activities, target=y_train_tensor,
            lr=lr_activity, m=m, lambda_target=lambda_target
        )
        model.update_weight(X_train_tensor, activities, y_train_tensor, lr=lr_weight)
    
    training_time = time.time() - start_time
    
    # 预测
    y_pred = model.predict(X_test_tensor, m=m)
    accuracy = accuracy_score(y_test, y_pred.cpu().numpy())
    
    return accuracy, training_time, y_pred.cpu().numpy()

# --------------------------
# 5. 基准测试函数
# --------------------------
def benchmark_methods(dataset_name, X_train, X_test, y_train, y_test, device='cpu'):
    """对比不同方法的性能"""
    results = {}
    
    print(f"\n测试数据集: {dataset_name.upper()}")
    print("=" * 50)
    
    # 1. discPC方法
    print("训练 discPC...")
    if dataset_name == 'wine':
        layers = [13, 32, 16, 3]
        n_epochs = 2000
    elif dataset_name == 'breast_cancer':
        layers = [30, 64, 32, 2]
        n_epochs = 3000
    elif dataset_name == 'digits':
        layers = [64, 128, 64, 10]
        n_epochs = 5000
    else:
        layers = [X_train.shape[1], 32, 16, len(np.unique(y_train))]
        n_epochs = 1000
    
    discpc_acc, discpc_time, discpc_pred = train_discpc(
        X_train, X_test, y_train, y_test, layers, n_epochs=n_epochs, device=device
    )
    results['discPC'] = {'accuracy': discpc_acc, 'time': discpc_time}
    print(f"discPC - 准确率: {discpc_acc:.4f}, 时间: {discpc_time:.2f}s")
    
    # 2. 传统机器学习方法
    methods = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'SVM (Linear)': SVC(kernel='linear', random_state=42),
        'MLP (sklearn)': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
    }
    
    for method_name, method in methods.items():
        print(f"训练 {method_name}...")
        start_time = time.time()
        
        # 训练
        method.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # 预测
        y_pred = method.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[method_name] = {'accuracy': accuracy, 'time': training_time}
        print(f"{method_name} - 准确率: {accuracy:.4f}, 时间: {training_time:.2f}s")
    
    return results

# --------------------------
# 6. 可视化结果
# --------------------------
def plot_benchmark_results(all_results, save_path=None):
    """绘制基准测试结果"""
    # 准备数据
    datasets = list(all_results.keys())
    methods = list(all_results[datasets[0]].keys())
    
    # 准确率对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 准确率柱状图
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        accuracies = [all_results[dataset][method]['accuracy'] for dataset in datasets]
        ax1.bar(x + i * width, accuracies, width, label=method, alpha=0.8)
    
    ax1.set_xlabel('数据集')
    ax1.set_ylabel('准确率')
    ax1.set_title('不同方法在各数据集上的准确率对比')
    ax1.set_xticks(x + width * (len(methods) - 1) / 2)
    ax1.set_xticklabels([d.upper() for d in datasets])
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 训练时间对比（对数尺度）
    for i, method in enumerate(methods):
        times = [all_results[dataset][method]['time'] for dataset in datasets]
        ax2.bar(x + i * width, times, width, label=method, alpha=0.8)
    
    ax2.set_xlabel('数据集')
    ax2.set_ylabel('训练时间 (秒)')
    ax2.set_title('不同方法的训练时间对比')
    ax2.set_yscale('log')
    ax2.set_xticks(x + width * (len(methods) - 1) / 2)
    ax2.set_xticklabels([d.upper() for d in datasets])
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_results_table(all_results):
    """创建结果表格"""
    datasets = list(all_results.keys())
    methods = list(all_results[datasets[0]].keys())
    
    # 创建DataFrame
    data = []
    for dataset in datasets:
        for method in methods:
            result = all_results[dataset][method]
            data.append({
                '数据集': dataset.upper(),
                '方法': method,
                '准确率': f"{result['accuracy']:.4f}",
                '训练时间(秒)': f"{result['time']:.2f}"
            })
    
    df = pd.DataFrame(data)
    return df

# --------------------------
# 7. 主函数
# --------------------------
def run_benchmark():
    """运行完整的基准测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试数据集
    datasets = ['wine', 'breast_cancer', 'digits']
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"开始测试数据集: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # 加载数据
        X_train, X_test, y_train, y_test, class_names, scaler = load_dataset_for_benchmark(dataset_name)
        print(f"数据加载完成 - 训练样本: {X_train.shape[0]}, 测试样本: {X_test.shape[0]}")
        print(f"特征维度: {X_train.shape[1]}, 类别数: {len(np.unique(y_train))}")
        
        # 运行基准测试
        results = benchmark_methods(dataset_name, X_train, X_test, y_train, y_test, device)
        all_results[dataset_name] = results
    
    # 绘制结果
    plot_benchmark_results(all_results, 'benchmark_results.png')
    
    # 创建结果表格
    results_df = create_results_table(all_results)
    print(f"\n{'='*80}")
    print("基准测试结果汇总")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    # 保存结果到CSV
    results_df.to_csv('benchmark_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到 benchmark_results.csv")
    
    # 分析discPC性能
    print(f"\n{'='*60}")
    print("discPC性能分析")
    print(f"{'='*60}")
    
    for dataset in datasets:
        discpc_acc = all_results[dataset]['discPC']['accuracy']
        discpc_time = all_results[dataset]['discPC']['time']
        
        # 找到最佳传统方法
        best_traditional_acc = 0
        best_traditional_method = ""
        for method, result in all_results[dataset].items():
            if method != 'discPC' and result['accuracy'] > best_traditional_acc:
                best_traditional_acc = result['accuracy']
                best_traditional_method = method
        
        print(f"\n{dataset.upper()}:")
        print(f"  discPC准确率: {discpc_acc:.4f}")
        print(f"  最佳传统方法: {best_traditional_method} ({best_traditional_acc:.4f})")
        print(f"  性能差距: {discpc_acc - best_traditional_acc:+.4f}")
        print(f"  训练时间: {discpc_time:.2f}秒")
    
    return all_results

# --------------------------
# 8. 运行基准测试
# --------------------------
if __name__ == "__main__":
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行基准测试
    results = run_benchmark()
    
    print(f"\n基准测试完成！discPC与传统方法进行了全面对比。")
