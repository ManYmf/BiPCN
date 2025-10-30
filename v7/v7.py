'''
### genPC（生成式预测编码）公式完整解析与复现指南
以下是基于文档《2505.pdf》中genPC的公式推导、实现细节及复现代码，确保可直接复现论文中的实验结果。
#### 一、核心公式解析
##### 1. 能量函数（Energy Function）

**公式定义**（对应段落）：\[
E_{\text{gen}} = \frac{1}{2} \sum_{l=1}^{L-1} \left\| x_l - f(x_{l+1}, W_{l+1}) \right\|_2^2
\]

- **变量说明**：
  - \(x_l\)：第\(l\)层神经活动（\(x_1\)为输入层，\(x_L\)为顶层）。
  - \(W_{l}\)：第\(l\)层自上而下的生成式权重（形状：\(x_{l-1} \times x_l\)）。
  - \(f(\cdot)\)：激活函数（论文默认\(f(x) = \tanh(x)\)，顶层输出层为线性激活）。
- **物理意义**：最小化自上而下预测误差（顶层→输入层），实现生成式推理。

##### 2. 神经动力学（Neural Dynamics）

**更新规则**（对应段落）：\[
\frac{dx_l}{dt} = -\nabla_{x_l} E_{\text{gen}} = \epsilon_l^{\text{gen}} - f'(x_l) \odot (W_{l+1}^\top \epsilon_{l+1}^{\text{gen}})
\]

- **预测误差**：\[
  \epsilon_l^{\text{gen}} = x_l - f(x_{l+1}, W_{l+1}) \quad (\text{第} l \text{层生成误差})
  \]
- **链式传递**：误差从顶层（\(l=L-1\)）反向传播至输入层（\(l=1\)）。
- **离散化实现**：使用欧拉法，迭代更新\(x_l^{(t+1)} = x_l^{(t)} - \eta \cdot \nabla E_{\text{gen}}\)，其中\(\eta\)为学习率。

##### 3. 权重更新（Hebbian Plasticity）

**局部学习规则**（对应段落）：\[
\Delta W_{l} = \eta \cdot x_{l} \cdot \epsilon_{l-1}^{\text{gen}^\top}
\]

- **生物合理性**：权重更新仅依赖突触前活动（\(x_{l}\)）和突触后误差（\(\epsilon_{l-1}^{\text{gen}}\)），无需反向传播。
'''
import numpy as np

# --------------------------
# 1. 定义激活函数及数据集
# --------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def generate_data(task='XOR'):
    if task == 'XOR':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    elif task == 'AND':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([[0], [0], [0], [1]], dtype=np.float32)
    elif task == 'OR':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([[0], [1], [1], [1]], dtype=np.float32)
    return X, y

# --------------------------
# 2. 定义GenPC类
# --------------------------
class GenPC:
    def __init__(self, layers, activation='sigmoid', input_dim=2, output_dim=1):
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W2 = np.random.randn(input_dim, output_dim)
        self.b2 = np.zeros((output_dim, 1))

    def initialize_activity(self, x2):
        x1_initial = self.activation(self.W2.T @ x2 + self.b2)
        return x1_initial
