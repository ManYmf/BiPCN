import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import itertools
import time
import pandas as pd
from tqdm import tqdm

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
# 2. discPC网络类（用于超参数调优）
# --------------------------
class discPC_Tuning:
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
def load_dataset_for_tuning(dataset_name, test_size=0.2):
    """加载数据集用于超参数调优"""
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
    
    return X_train, X_test, y_train, y_test, class_names

# --------------------------
# 4. 超参数调优函数
# --------------------------
def tune_hyperparameters(dataset_name, X_train, X_test, y_train, y_test, device='cpu'):
    """对指定数据集进行超参数调优"""
    print(f"\n开始调优 {dataset_name.upper()} 数据集的超参数...")
    
    # 定义超参数搜索空间
    if dataset_name == 'wine':
        param_grid = {
            'layers': [
                [13, 16, 3],
                [13, 32, 3],
                [13, 64, 3],
                [13, 32, 16, 3],
                [13, 64, 32, 3],
                [13, 128, 64, 3]
            ],
            'n_epochs': [1000, 2000, 3000],
            'm': [5, 10, 15, 20],
            'lr_activity': [0.01, 0.03, 0.05, 0.1],
            'lr_weight': [0.01, 0.03, 0.05, 0.1],
            'lambda_target': [0.5, 0.7, 0.8, 0.9]
        }
    elif dataset_name == 'breast_cancer':
        param_grid = {
            'layers': [
                [30, 16, 2],
                [30, 32, 2],
                [30, 64, 2],
                [30, 32, 16, 2],
                [30, 64, 32, 2],
                [30, 128, 64, 2]
            ],
            'n_epochs': [2000, 3000, 5000],
            'm': [10, 15, 20, 25],
            'lr_activity': [0.01, 0.02, 0.05, 0.1],
            'lr_weight': [0.01, 0.02, 0.05, 0.1],
            'lambda_target': [0.6, 0.7, 0.8, 0.9]
        }
    elif dataset_name == 'digits':
        param_grid = {
            'layers': [
                [64, 32, 10],
                [64, 64, 10],
                [64, 128, 10],
                [64, 64, 32, 10],
                [64, 128, 64, 10],
                [64, 256, 128, 10]
            ],
            'n_epochs': [3000, 5000, 8000],
            'm': [15, 20, 25, 30],
            'lr_activity': [0.005, 0.01, 0.02, 0.05],
            'lr_weight': [0.005, 0.01, 0.02, 0.05],
            'lambda_target': [0.7, 0.8, 0.9, 0.95]
        }
    
    # 生成参数组合（限制数量以避免计算时间过长）
    param_combinations = list(ParameterGrid(param_grid))
    
    # 如果组合太多，随机采样
    if len(param_combinations) > 100:
        np.random.seed(42)
        param_combinations = np.random.choice(param_combinations, 100, replace=False).tolist()
    
    print(f"总共测试 {len(param_combinations)} 种参数组合...")
    
    # 存储结果
    results = []
    
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device).T
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device).T
    
    # 独热编码标签
    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_encoded = encoder.transform(y_test.reshape(-1, 1))
    
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.float32, device=device).T
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.float32, device=device).T
    
    # 遍历参数组合
    for i, params in enumerate(tqdm(param_combinations, desc="调优进度")):
        try:
            # 创建模型
            model = discPC_Tuning(params['layers'], device=device)
            
            # 训练
            start_time = time.time()
            
            for epoch in range(params['n_epochs']):
                activities = model.initialize_activity(X_train_tensor)
                activities = model.update_activity(
                    X_train_tensor, activities, target=y_train_tensor,
                    lr=params['lr_activity'], m=params['m'], 
                    lambda_target=params['lambda_target']
                )
                model.update_weight(X_train_tensor, activities, y_train_tensor, 
                                  lr=params['lr_weight'])
            
            training_time = time.time() - start_time
            
            # 预测
            y_pred = model.predict(X_test_tensor, m=params['m'])
            accuracy = accuracy_score(y_test, y_pred.cpu().numpy())
            
            # 记录结果
            result = params.copy()
            result['accuracy'] = accuracy
            result['training_time'] = training_time
            results.append(result)
            
        except Exception as e:
            print(f"参数组合 {i+1} 失败: {e}")
            continue
    
    # 转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    return results_df

# --------------------------
# 5. 可视化调优结果
# --------------------------
def plot_tuning_results(results_df, dataset_name, top_n=10):
    """可视化超参数调优结果"""
    top_results = results_df.head(top_n)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{dataset_name.upper()} 超参数调优结果 (Top {top_n})', fontsize=16)
    
    # 1. 准确率分布
    axes[0, 0].hist(results_df['accuracy'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(top_results['accuracy'].iloc[0], color='red', linestyle='--', 
                      label=f'最佳: {top_results["accuracy"].iloc[0]:.4f}')
    axes[0, 0].set_xlabel('准确率')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('准确率分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 学习率 vs 准确率
    axes[0, 1].scatter(top_results['lr_activity'], top_results['accuracy'], 
                      c=top_results['lr_weight'], cmap='viridis', alpha=0.7)
    axes[0, 1].set_xlabel('活动学习率')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].set_title('学习率 vs 准确率')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 迭代次数 vs 准确率
    axes[0, 2].scatter(top_results['m'], top_results['accuracy'], 
                      c=top_results['n_epochs'], cmap='plasma', alpha=0.7)
    axes[0, 2].set_xlabel('活动更新次数 (m)')
    axes[0, 2].set_ylabel('准确率')
    axes[0, 2].set_title('迭代次数 vs 准确率')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. lambda_target vs 准确率
    axes[1, 0].scatter(top_results['lambda_target'], top_results['accuracy'], 
                      c=top_results['accuracy'], cmap='coolwarm', alpha=0.7)
    axes[1, 0].set_xlabel('Lambda Target')
    axes[1, 0].set_ylabel('准确率')
    axes[1, 0].set_title('Lambda Target vs 准确率')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 训练时间 vs 准确率
    axes[1, 1].scatter(top_results['training_time'], top_results['accuracy'], 
                      alpha=0.7, color='green')
    axes[1, 1].set_xlabel('训练时间 (秒)')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].set_title('训练时间 vs 准确率')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 网络层数 vs 准确率
    layer_counts = [len(layers) for layers in top_results['layers']]
    axes[1, 2].scatter(layer_counts, top_results['accuracy'], 
                      alpha=0.7, color='purple')
    axes[1, 2].set_xlabel('网络层数')
    axes[1, 2].set_ylabel('准确率')
    axes[1, 2].set_title('网络层数 vs 准确率')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_parameter_importance(results_df, dataset_name):
    """分析参数重要性"""
    # 计算参数与准确率的相关性
    numeric_cols = ['n_epochs', 'm', 'lr_activity', 'lr_weight', 'lambda_target', 'training_time']
    correlations = results_df[numeric_cols + ['accuracy']].corr()['accuracy'].drop('accuracy')
    
    # 绘制相关性图
    plt.figure(figsize=(10, 6))
    correlations.plot(kind='bar', color='skyblue', alpha=0.7)
    plt.title(f'{dataset_name.upper()} - 参数与准确率的相关性')
    plt.xlabel('参数')
    plt.ylabel('相关系数')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return correlations

# --------------------------
# 6. 主函数
# --------------------------
def run_hyperparameter_tuning():
    """运行超参数调优"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试数据集
    datasets = ['wine', 'breast_cancer', 'digits']
    all_tuning_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"开始超参数调优: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # 加载数据
        X_train, X_test, y_train, y_test, class_names = load_dataset_for_tuning(dataset_name)
        print(f"数据加载完成 - 训练样本: {X_train.shape[0]}, 测试样本: {X_test.shape[0]}")
        print(f"特征维度: {X_train.shape[1]}, 类别数: {len(np.unique(y_train))}")
        
        # 运行超参数调优
        tuning_results = tune_hyperparameters(dataset_name, X_train, X_test, y_train, y_test, device)
        all_tuning_results[dataset_name] = tuning_results
        
        # 显示最佳结果
        print(f"\n{dataset_name.upper()} 最佳配置:")
        print("=" * 50)
        best_config = tuning_results.iloc[0]
        print(f"准确率: {best_config['accuracy']:.4f}")
        print(f"网络结构: {best_config['layers']}")
        print(f"训练轮数: {best_config['n_epochs']}")
        print(f"活动更新次数: {best_config['m']}")
        print(f"活动学习率: {best_config['lr_activity']}")
        print(f"权重学习率: {best_config['lr_weight']}")
        print(f"Lambda Target: {best_config['lambda_target']}")
        print(f"训练时间: {best_config['training_time']:.2f}秒")
        
        # 可视化结果
        plot_tuning_results(tuning_results, dataset_name)
        correlations = plot_parameter_importance(tuning_results, dataset_name)
        
        # 保存结果
        tuning_results.to_csv(f'{dataset_name}_tuning_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n{dataset_name.upper()} 调优结果已保存到 {dataset_name}_tuning_results.csv")
    
    # 总结最佳配置
    print(f"\n{'='*80}")
    print("超参数调优总结 - 最佳配置")
    print(f"{'='*80}")
    
    summary_data = []
    for dataset_name, results in all_tuning_results.items():
        best = results.iloc[0]
        summary_data.append({
            '数据集': dataset_name.upper(),
            '最佳准确率': f"{best['accuracy']:.4f}",
            '网络结构': str(best['layers']),
            '训练轮数': best['n_epochs'],
            '活动更新次数': best['m'],
            '活动学习率': best['lr_activity'],
            '权重学习率': best['lr_weight'],
            'Lambda Target': best['lambda_target'],
            '训练时间(秒)': f"{best['training_time']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # 保存总结
    summary_df.to_csv('hyperparameter_tuning_summary.csv', index=False, encoding='utf-8-sig')
    print(f"\n调优总结已保存到 hyperparameter_tuning_summary.csv")
    
    return all_tuning_results

# --------------------------
# 7. 运行超参数调优
# --------------------------
if __name__ == "__main__":
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行超参数调优
    results = run_hyperparameter_tuning()
    
    print(f"\n超参数调优完成！已为每个数据集找到最佳配置。")
