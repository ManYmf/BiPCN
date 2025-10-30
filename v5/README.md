# discPC 复杂数据集测试

这个目录包含了discPC算法在多个复杂数据集上的测试代码，以及与传统机器学习方法的性能对比。

## 文件说明

### 核心文件

- `v5.py` - 原始IRIS数据集测试代码
- `v5_complex_datasets.py` - 复杂数据集测试（Wine、Breast Cancer、Digits等）
- `v5_benchmark.py` - 与sklearn传统方法的性能基准对比
- `v5_hyperparameter_tuning.py` - 超参数调优和最佳配置搜索
- `run_all_tests.py` - 综合测试脚本

### 辅助文件

- `compare.py` - 不同版本discPC的对比分析
- `README.md` - 本说明文件

## 快速开始

### 1. 运行复杂数据集测试

```bash
python v5_complex_datasets.py
```

这将测试discPC在Wine、Breast Cancer、Digits数据集上的性能，并生成训练曲线和混淆矩阵。

### 2. 运行性能基准对比

```bash
python v5_benchmark.py
```

这将对比discPC与Logistic Regression、Random Forest、SVM、MLP等传统方法的性能。

### 3. 运行超参数调优

```bash
python v5_hyperparameter_tuning.py
```

这将为每个数据集搜索最佳的超参数配置。

### 4. 运行所有测试

```bash
python run_all_tests.py
```

或者选择特定测试：

```bash
python run_all_tests.py --test complex
python run_all_tests.py --test benchmark
python run_all_tests.py --test tuning
```

## 测试数据集

### Wine数据集

- **特征数**: 13
- **样本数**: 178
- **类别数**: 3
- **难度**: 中等
- **特点**: 多特征，类别平衡

### Breast Cancer数据集

- **特征数**: 30
- **样本数**: 569
- **类别数**: 2
- **难度**: 中等
- **特点**: 高维特征，二分类

### Digits数据集

- **特征数**: 64
- **样本数**: 1797
- **类别数**: 10
- **难度**: 较高
- **特点**: 图像数据，多类别分类

## 网络架构配置

### Wine数据集

```python
layers = [13, 32, 16, 3]  # 输入层13个，隐藏层32和16个，输出层3个
```

### Breast Cancer数据集

```python
layers = [30, 64, 32, 2]  # 输入层30个，隐藏层64和32个，输出层2个
```

### Digits数据集

```python
layers = [64, 128, 64, 10]  # 输入层64个，隐藏层128和64个，输出层10个
```

## 超参数说明

### 主要超参数

- `n_epochs`: 训练轮数（1000-8000）
- `m`: 活动值更新次数（5-30）
- `lr_activity`: 活动值学习率（0.005-0.1）
- `lr_weight`: 权重学习率（0.005-0.1）
- `lambda_target`: 目标误差权重（0.5-0.95）

### 网络结构参数

- `layers`: 各层神经元数量列表
- `use_bias`: 是否使用偏置项
- `weight_init`: 权重初始化方法（'xavier', 'he', 'normal'）
- `dropout_rate`: Dropout率（0-0.5）

## 性能指标

### 评估指标

- **准确率**: 分类正确的样本比例
- **训练时间**: 模型训练所需时间
- **收敛速度**: 达到稳定性能的轮数

### 对比方法

- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM (RBF)
- SVM (Linear)
- MLP (sklearn)

## 结果文件

### 自动生成的文件

- `benchmark_results.csv` - 基准测试结果
- `benchmark_results.png` - 性能对比图表
- `*_tuning_results.csv` - 各数据集超参数调优结果
- `hyperparameter_tuning_summary.csv` - 调优总结

### 可视化输出

- 训练曲线图
- 混淆矩阵
- 参数重要性分析
- 性能对比柱状图

## 使用建议

### 1. 首次运行

建议先运行 `v5_complex_datasets.py` 来了解discPC在不同数据集上的基本性能。

### 2. 性能对比

运行 `v5_benchmark.py` 来了解discPC相对于传统方法的优势。

### 3. 参数优化

如果对性能不满意，可以运行 `v5_hyperparameter_tuning.py` 来寻找最佳配置。

### 4. 快速测试

使用 `run_all_tests.py --quick` 进行快速测试，减少参数组合数量。

## 注意事项

### 计算资源

- GPU加速：如果可用，会自动使用CUDA
- 内存需求：Digits数据集需要较多内存
- 时间成本：超参数调优可能需要较长时间

### 依赖包

确保安装以下Python包：

```bash
pip install torch numpy scikit-learn matplotlib seaborn pandas tqdm
```

### 随机种子

所有测试都设置了随机种子（seed=0），确保结果可复现。

## 故障排除

### 常见问题

1. **内存不足**: 减少网络层数或使用CPU
2. **训练时间过长**: 减少epochs或使用快速模式
3. **准确率较低**: 尝试不同的超参数配置

### 调试建议

- 检查数据预处理是否正确
- 验证网络架构是否合理
- 调整学习率和迭代次数
- 检查激活函数选择

## 扩展功能

### 添加新数据集

1. 在 `load_dataset` 函数中添加新数据集
2. 在 `datasets_config` 中配置网络结构
3. 调整超参数搜索空间

### 自定义网络架构

1. 修改 `discPC_Enhanced` 类
2. 添加新的激活函数
3. 实现新的权重初始化方法

### 性能优化

1. 使用更高效的优化算法
2. 实现批处理训练
3. 添加早停机制

## 联系信息

如有问题或建议，请查看代码注释或相关文档。
