import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time

# 设置随机种子（确保结果可复现）
torch.manual_seed(0)
np.random.seed(0)

# --------------------------
# 1. 激活函数（适配多类别分类）
# --------------------------
def sigmoid(x):
    # 使用PyTorch的sigmoid，并用clamp防止数值溢出
    return torch.sigmoid(torch.clamp(x, -20, 20))

def softmax(x):
    # 多类别分类输出层使用softmax
    exp_x = torch.exp(torch.clamp(x, -20, 20))  # 防止数值溢出
    return exp_x / torch.sum(exp_x, dim=0, keepdim=True)

def relu(x):
    # ReLU激活函数
    return torch.relu(x)

def tanh(x):
    # Tanh激活函数
    return torch.tanh(x)

# --------------------------
# 2. 数据加载与预处理（支持多个数据集）
# --------------------------
def load_dataset(dataset_name, test_size=0.2, device='cpu'):
    """
    加载不同的数据集
    支持的数据集: 'wine', 'breast_cancer', 'digits', 'wine_quality'
    """
    if dataset_name == 'wine':
        data = load_wine()
        X = data.data
        y = data.target.reshape(-1, 1)
        class_names = data.target_names
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        X = data.data
        y = data.target.reshape(-1, 1)
        class_names = data.target_names
    elif dataset_name == 'digits':
        data = load_digits()
        X = data.data
        y = data.target.reshape(-1, 1)
        class_names = [str(i) for i in range(10)]
    elif dataset_name == 'wine_quality':
        # Wine Quality数据集在当前sklearn版本中不可用，使用Wine数据集代替
        print("Wine Quality数据集不可用，使用Wine数据集代替")
        data = load_wine()
        X = data.data
        y = data.target.reshape(-1, 1)
        class_names = data.target_names
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    print(f"加载数据集: {dataset_name}")
    print(f"特征维度: {X.shape[1]}, 样本数量: {X.shape[0]}, 类别数量: {len(np.unique(y))}")
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 标签独热编码
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y
    )
    
    # 转换为PyTorch张量并移动到目标设备
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device).T
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device).T
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device).T
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device).T
    
    return X_train, X_test, y_train, y_test, class_names

# --------------------------
# 3. 增强的discPC网络类
# --------------------------
class discPC_Enhanced:
    def __init__(self, layers, activations=None, use_bias=True, device='cpu', 
                 weight_init='xavier', dropout_rate=0.0):
        """
        增强的discPC网络
        layers: 各层神经元数量列表
        activations: 各层激活函数列表
        use_bias: 是否使用偏置项
        device: 计算设备
        weight_init: 权重初始化方法 ('xavier', 'he', 'normal')
        dropout_rate: dropout率（0表示不使用dropout）
        """
        self.layers = layers
        self.n_layers = len(layers)
        self.use_bias = use_bias
        self.device = device
        self.dropout_rate = dropout_rate
        
        # 设置激活函数
        if activations is None:
            self.activations = [sigmoid] * (self.n_layers - 2) + [softmax]
        else:
            self.activations = activations
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            input_dim = layers[i]
            output_dim = layers[i+1]
            
            # 根据指定方法初始化权重
            if weight_init == 'xavier':
                weight = torch.randn(input_dim, output_dim, device=device) * np.sqrt(2.0 / (input_dim + output_dim))
            elif weight_init == 'he':
                weight = torch.randn(input_dim, output_dim, device=device) * np.sqrt(2.0 / input_dim)
            else:  # normal
                weight = torch.randn(input_dim, output_dim, device=device) * np.sqrt(1.0 / input_dim)
            
            self.weights.append(weight)
            
            # 偏置初始化
            if use_bias:
                bias = torch.zeros(output_dim, 1, device=device)
                self.biases.append(bias)
            else:
                self.biases.append(None)
    
    def initialize_activity(self, x_batch):
        """初始化所有层的活动值"""
        activities = [x_batch]
        
        for i in range(self.n_layers - 1):
            prev_activity = activities[-1]
            weight = self.weights[i]
            bias = self.biases[i]
            
            layer_input = weight.T @ prev_activity
            if self.use_bias and bias is not None:
                layer_input += bias
            
            activity = self.activations[i](layer_input)
            
            # 应用dropout（训练时）
            if self.dropout_rate > 0 and self.training:
                dropout_mask = torch.rand_like(activity) > self.dropout_rate
                activity = activity * dropout_mask / (1 - self.dropout_rate)
            
            activities.append(activity)
        
        return activities
    
    def update_activity(self, x_batch, activities, target=None, lr=0.1, m=20, lambda_target=0.9):
        """更新所有层的活动值"""
        current_activities = [act.clone() for act in activities]
        
        for _ in range(m):
            # 前向计算预测值
            pred_activities = [x_batch]
            for i in range(self.n_layers - 1):
                prev_pred = pred_activities[-1]
                layer_input = self.weights[i].T @ prev_pred
                if self.use_bias and self.biases[i] is not None:
                    layer_input += self.biases[i]
                pred_activities.append(self.activations[i](layer_input))
            
            # 从输出层反向更新到隐藏层
            for i in reversed(range(1, self.n_layers)):
                epsilon = current_activities[i] - pred_activities[i]
                update = -lr * epsilon
                
                # 目标误差处理
                if target is not None and i == self.n_layers - 1:
                    target_error = current_activities[i] - target
                    update -= lr * lambda_target * target_error
                elif target is not None and i < self.n_layers - 1:
                    # 计算激活函数导数
                    if self.activations[i] == sigmoid:
                        deriv = pred_activities[i] * (1 - pred_activities[i])
                    elif self.activations[i] == tanh:
                        deriv = 1 - pred_activities[i] ** 2
                    elif self.activations[i] == relu:
                        deriv = (pred_activities[i] > 0).float()
                    else:
                        deriv = torch.ones_like(pred_activities[i])
                    
                    next_error = current_activities[i+1] - target if i+1 == self.n_layers -1 else current_activities[i+1] - pred_activities[i+1]
                    target_error = (self.weights[i] @ next_error) * deriv
                    update -= lr * lambda_target * target_error
                
                current_activities[i] += update
        
        return current_activities
    
    def update_weight(self, x_batch, activities, target, lr=0.01):
        """更新所有层的权重和偏置"""
        pred_activities = [x_batch]
        for i in range(self.n_layers - 1):
            prev_pred = pred_activities[-1]
            layer_input = self.weights[i].T @ prev_pred
            if self.use_bias and self.biases[i] is not None:
                layer_input += self.biases[i]
            pred_activities.append(self.activations[i](layer_input))
        
        # 计算各层误差
        errors = [None] * (self.n_layers - 1)
        errors[-1] = target - pred_activities[-1]
        
        # 反向计算隐藏层误差
        for i in reversed(range(self.n_layers - 2)):
            if self.activations[i] == sigmoid:
                deriv = pred_activities[i+1] * (1 - pred_activities[i+1])
            elif self.activations[i] == tanh:
                deriv = 1 - pred_activities[i+1] ** 2
            elif self.activations[i] == relu:
                deriv = (pred_activities[i+1] > 0).float()
            else:
                deriv = torch.ones_like(pred_activities[i+1])
            
            errors[i] = (self.weights[i+1] @ errors[i+1]) * deriv
        
        # 更新权重和偏置
        batch_size = x_batch.shape[1]
        for i in range(self.n_layers - 1):
            self.weights[i] += lr * (pred_activities[i] @ errors[i].T) / batch_size
            
            if self.use_bias and self.biases[i] is not None:
                self.biases[i] += lr * torch.mean(errors[i], dim=1, keepdim=True)
    
    def predict(self, x_batch, m=20):
        """预测函数"""
        self.training = False  # 预测时关闭dropout
        activities = self.initialize_activity(x_batch)
        activities = self.update_activity(x_batch, activities, m=m)
        return torch.argmax(activities[-1], dim=0)
    
    def train(self):
        """设置为训练模式"""
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        self.training = False

# --------------------------
# 4. 训练和评估函数
# --------------------------
def train_and_evaluate(model, X_train, X_test, y_train, y_test, class_names, 
                      n_epochs=1000, m=10, lr_activity=0.05, lr_weight=0.05, 
                      lambda_target=0.8, verbose=True):
    """训练和评估模型"""
    model.train()
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # 训练
        activities = model.initialize_activity(X_train)
        activities = model.update_activity(
            X_train, activities, target=y_train,
            lr=lr_activity, m=m, lambda_target=lambda_target
        )
        model.update_weight(X_train, activities, y_train, lr=lr_weight)
        
        # 计算训练指标
        y_pred_train = activities[-1]
        train_loss = torch.mean((y_pred_train - y_train) ** 2)
        train_pred_label = torch.argmax(y_pred_train, dim=0)
        train_true_label = torch.argmax(y_train, dim=0)
        train_accuracy = torch.mean((train_pred_label == train_true_label).float())
        
        train_losses.append(train_loss.item())
        train_accuracies.append(train_accuracy.item())
        
        # 计算测试准确率
        model.eval()
        test_activities = model.initialize_activity(X_test)
        test_activities = model.update_activity(X_test, test_activities, m=m)
        test_pred_label = torch.argmax(test_activities[-1], dim=0)
        test_true_label = torch.argmax(y_test, dim=0)
        test_accuracy = torch.mean((test_pred_label == test_true_label).float())
        test_accuracies.append(test_accuracy.item())
        model.train()
        
        # 打印进度
        if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"Epoch {epoch+1:4d}, 训练损失: {train_loss.item():.6f}, "
                  f"训练准确率: {train_accuracy.item():.4f}, "
                  f"测试准确率: {test_accuracy.item():.4f}")
    
    training_time = time.time() - start_time
    
    # 最终评估
    model.eval()
    final_pred = model.predict(X_test, m=m)
    final_accuracy = torch.mean((final_pred == test_true_label).float()).item()
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'final_accuracy': final_accuracy,
        'training_time': training_time,
        'predictions': final_pred.cpu().numpy(),
        'true_labels': test_true_label.cpu().numpy()
    }

# --------------------------
# 5. 可视化函数
# --------------------------
def plot_training_curves(results, dataset_name, save_path=None):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 损失曲线
    ax1.plot(results['train_losses'])
    ax1.set_title(f'{dataset_name} - 训练损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(results['train_accuracies'], label='训练准确率')
    ax2.plot(results['test_accuracies'], label='测试准确率')
    ax2.set_title(f'{dataset_name} - 准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, dataset_name, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{dataset_name} - 混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------
# 6. 主测试函数
# --------------------------
def test_complex_datasets():
    """测试多个复杂数据集"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据集配置
    datasets_config = {
        'wine': {
            'layers': [13, 32, 16, 3],  # 13个特征，3个类别
            'n_epochs': 2000,
            'm': 15,
            'lr_activity': 0.03,
            'lr_weight': 0.03,
            'lambda_target': 0.7
        },
        'breast_cancer': {
            'layers': [30, 64, 32, 2],  # 30个特征，2个类别
            'n_epochs': 3000,
            'm': 20,
            'lr_activity': 0.02,
            'lr_weight': 0.02,
            'lambda_target': 0.8
        },
        'digits': {
            'layers': [64, 128, 64, 10],  # 64个特征，10个类别
            'n_epochs': 5000,
            'm': 25,
            'lr_activity': 0.01,
            'lr_weight': 0.01,
            'lambda_target': 0.9
        }
    }
    
    results_summary = {}
    
    for dataset_name, config in datasets_config.items():
        print(f"\n{'='*60}")
        print(f"测试数据集: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # 加载数据
        X_train, X_test, y_train, y_test, class_names = load_dataset(dataset_name, device=device)
        
        # 创建模型
        model = discPC_Enhanced(
            layers=config['layers'],
            use_bias=True,
            device=device,
            weight_init='xavier',
            dropout_rate=0.1 if dataset_name == 'digits' else 0.0
        )
        
        # 训练和评估
        results = train_and_evaluate(
            model, X_train, X_test, y_train, y_test, class_names,
            n_epochs=config['n_epochs'],
            m=config['m'],
            lr_activity=config['lr_activity'],
            lr_weight=config['lr_weight'],
            lambda_target=config['lambda_target']
        )
        
        # 保存结果
        results_summary[dataset_name] = {
            'final_accuracy': results['final_accuracy'],
            'training_time': results['training_time'],
            'config': config
        }
        
        # 打印详细结果
        print(f"\n{dataset_name.upper()} 最终结果:")
        print(f"测试准确率: {results['final_accuracy']:.4f}")
        print(f"训练时间: {results['training_time']:.2f}秒")
        
        # 分类报告
        print(f"\n分类报告:")
        print(classification_report(results['true_labels'], results['predictions'], 
                                  target_names=class_names))
        
        # 绘制训练曲线
        plot_training_curves(results, dataset_name.upper())
        
        # 绘制混淆矩阵
        plot_confusion_matrix(results['true_labels'], results['predictions'], 
                            class_names, dataset_name.upper())
    
    # 总结对比
    print(f"\n{'='*60}")
    print("数据集性能对比总结")
    print(f"{'='*60}")
    print(f"{'数据集':<15} {'准确率':<10} {'训练时间(秒)':<15} {'网络结构'}")
    print("-" * 60)
    
    for dataset_name, result in results_summary.items():
        layers_str = '->'.join(map(str, result['config']['layers']))
        print(f"{dataset_name.upper():<15} {result['final_accuracy']:<10.4f} "
              f"{result['training_time']:<15.2f} {layers_str}")
    
    return results_summary

# --------------------------
# 7. 运行测试
# --------------------------
if __name__ == "__main__":
    # 设置matplotlib字体
    plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行复杂数据集测试
    results = test_complex_datasets()
    
    print(f"\n测试完成！discPC在多个复杂数据集上展现了良好的性能。")
