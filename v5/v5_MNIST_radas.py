# v5_MNIST_trainable.py
# 改造点：
# 1) 增加 trainable(config) 作为唯一训练入口（config 传参，results 返回）
# 2) 保留原有模型/数据处理/训练逻辑，尽量少改
# 3) 可选保留 __main__ 便于本地直接跑（不会影响被平台调用 trainable）

import os
import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader


# --------------------------
# 1. 激活函数（保持原有，适配多类别）
# --------------------------
def sigmoid(x):
    return torch.sigmoid(torch.clamp(x, -20, 20))


def softmax(x):
    exp_x = torch.exp(torch.clamp(x, -20, 20))
    return exp_x / torch.sum(exp_x, dim=0, keepdim=True)


# --------------------------
# 2. 数据加载与预处理（适配MNIST）
# --------------------------
def load_mnist_data(
    batch_size=64,
    device="cpu",
    data_root="./data",
    download=True,
    num_workers=0,
    pin_memory=False,
):
    # 图像预处理：转为张量+归一化
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST均值和标准差
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=download, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=download, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 独热编码器（10个类别）
    # 兼容不同 sklearn 版本：sparse_output 新版；旧版用 sparse
    try:
        encoder = OneHotEncoder(sparse_output=False, categories="auto")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, categories="auto")
    encoder.fit(np.arange(10).reshape(-1, 1))

    return train_loader, test_loader, encoder, device


# --------------------------
# 3. 多层discPC网络类（保持原有）
# --------------------------
class discPC_Flexible:
    def __init__(self, layers, activations=None, use_bias=True, device="cpu"):
        self.layers = layers
        self.n_layers = len(layers)
        self.use_bias = use_bias
        self.device = device

        if activations is None:
            self.activations = [sigmoid] * (self.n_layers - 2) + [softmax]
        else:
            self.activations = activations

        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        for i in range(self.n_layers - 1):
            input_dim = layers[i]
            output_dim = layers[i + 1]
            weight = torch.randn(input_dim, output_dim, device=device) * torch.sqrt(
                torch.tensor(1 / input_dim, device=device)
            )
            self.weights.append(weight)
            if use_bias:
                self.biases.append(torch.zeros(output_dim, 1, device=device))
            else:
                self.biases.append(None)

    def initialize_activity(self, x_batch):
        activities = [x_batch]
        for i in range(self.n_layers - 1):
            prev_activity = activities[-1]
            layer_input = self.weights[i].T @ prev_activity
            if self.use_bias and self.biases[i] is not None:
                layer_input += self.biases[i]
            activity = self.activations[i](layer_input)
            activities.append(activity)
        return activities

    def update_activity(
        self, x_batch, activities, target=None, lr=0.1, m=20, lambda_target=0.9
    ):
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

            # 反向更新活动值
            for i in reversed(range(1, self.n_layers)):
                epsilon = current_activities[i] - pred_activities[i]
                update = -lr * epsilon
                if target is not None and i == self.n_layers - 1:
                    target_error = current_activities[i] - target
                    update -= lr * lambda_target * target_error
                elif target is not None and i < self.n_layers - 1:
                    sigmoid_deriv = pred_activities[i] * (1 - pred_activities[i])
                    next_error = (
                        current_activities[i + 1] - target
                        if i + 1 == self.n_layers - 1
                        else current_activities[i + 1] - pred_activities[i + 1]
                    )
                    target_error = (self.weights[i] @ next_error) * sigmoid_deriv
                    update -= lr * lambda_target * target_error
                current_activities[i] += update
        return current_activities

    def update_weight(self, x_batch, activities, target, lr=0.01):
        # 前向计算预测值
        pred_activities = [x_batch]
        for i in range(self.n_layers - 1):
            prev_pred = pred_activities[-1]
            layer_input = self.weights[i].T @ prev_pred
            if self.use_bias and self.biases[i] is not None:
                layer_input += self.biases[i]
            pred_activities.append(self.activations[i](layer_input))

        # 计算误差
        errors = [None] * (self.n_layers - 1)
        errors[-1] = target - pred_activities[-1]
        for i in reversed(range(self.n_layers - 2)):
            sigmoid_deriv = pred_activities[i + 1] * (1 - pred_activities[i + 1])
            errors[i] = (self.weights[i + 1] @ errors[i + 1]) * sigmoid_deriv

        # 更新权重和偏置
        batch_size = x_batch.shape[1]
        for i in range(self.n_layers - 1):
            self.weights[i] += lr * (pred_activities[i] @ errors[i].T) / batch_size
            if self.use_bias and self.biases[i] is not None:
                self.biases[i] += lr * torch.mean(errors[i], dim=1, keepdim=True)

    def predict(self, x_batch, m=20):
        activities = self.initialize_activity(x_batch)
        activities = self.update_activity(x_batch, activities, m=m)
        return torch.argmax(activities[-1], dim=0)


# --------------------------
# 4. trainable(config)：核心入口
# --------------------------
def trainable(config: dict):
    """
    约定：
    - 输入：config (dict)
    - 输出：results (dict)，包含可记录/汇总的指标与必要元信息
    """
    results = {}

    # ---- config: 读取 & 默认值 ----
    seed = int(config.get("seed", 0))
    batch_size = int(config.get("batch_size", 64))
    n_epochs = int(config.get("epochs", 50))
    layers = config.get("layers", [784, 128, 64, 10])
    m = int(config.get("m", 15))
    lr_activity = float(config.get("lr_activity", 0.03))
    lr_weight = float(config.get("lr_weight", 0.03))
    lambda_target = float(config.get("lambda_target", 0.7))
    use_bias = bool(config.get("use_bias", True))

    data_root = config.get("data_root", "./data")
    download = bool(config.get("download", True))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", False))

    log_every = int(config.get("log_every", 5))
    report_per_class = bool(config.get("report_per_class", False))

    # device: 允许 config 指定；否则自动选择
    device_cfg = config.get("device", None)
    if device_cfg is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    # ---- 设置随机种子（可复现）----
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    results["device"] = str(device)
    results["torch.cuda.is_available"] = bool(torch.cuda.is_available())

    # 把关键 config 回填进 results，方便你后续汇总成 df
    results["config/seed"] = seed
    results["config/batch_size"] = batch_size
    results["config/epochs"] = n_epochs
    results["config/layers"] = str(layers)
    results["config/m"] = m
    results["config/lr_activity"] = lr_activity
    results["config/lr_weight"] = lr_weight
    results["config/lambda_target"] = lambda_target
    results["config/use_bias"] = use_bias

    # ---- 数据 ----
    train_loader, test_loader, encoder, _ = load_mnist_data(
        batch_size=batch_size,
        device=device,
        data_root=data_root,
        download=download,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ---- 模型 ----
    model = discPC_Flexible(layers=layers, use_bias=use_bias, device=device)

    # ---- 训练循环 ----
    for epoch in range(n_epochs):
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for data, targets in train_loader:
            x_batch = data.view(-1, 784).T.to(device)  # (784, B)

            y_np = encoder.transform(targets.cpu().numpy().reshape(-1, 1))
            y_batch = torch.tensor(y_np, dtype=torch.float32, device=device).T  # (10, B)

            activities = model.initialize_activity(x_batch)
            activities = model.update_activity(
                x_batch,
                activities,
                target=y_batch,
                lr=lr_activity,
                m=m,
                lambda_target=lambda_target,
            )
            model.update_weight(x_batch, activities, y_batch, lr=lr_weight)

            y_pred = activities[-1]
            batch_loss = torch.mean((y_pred - y_batch) ** 2)
            train_loss_sum += float(batch_loss.item())

            pred_labels = torch.argmax(y_pred, dim=0)
            true_labels = torch.argmax(y_batch, dim=0)
            train_correct += int(torch.sum(pred_labels == true_labels).item())
            train_total += int(targets.size(0))

        avg_train_loss = train_loss_sum / max(1, len(train_loader))
        train_acc = train_correct / max(1, train_total)

        # ---- 测试评估 ----
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                x_batch = data.view(-1, 784).T.to(device)
                y_np = encoder.transform(targets.cpu().numpy().reshape(-1, 1))
                y_batch = torch.tensor(y_np, dtype=torch.float32, device=device).T

                test_activities = model.initialize_activity(x_batch)
                test_activities = model.update_activity(x_batch, test_activities, m=m)
                pred_labels = torch.argmax(test_activities[-1], dim=0)
                true_labels = torch.argmax(y_batch, dim=0)

                test_correct += int(torch.sum(pred_labels == true_labels).item())
                test_total += int(targets.size(0))

        test_acc = test_correct / max(1, test_total)

        # ---- 记录 results（按你之前 df 的习惯：epoch_k/metric）----
        ep = epoch + 1
        results[f"epoch_{ep}/train_loss"] = avg_train_loss
        results[f"epoch_{ep}/train_acc"] = train_acc
        results[f"epoch_{ep}/test_acc"] = test_acc

        if log_every > 0 and (ep % log_every == 0):
            print(
                f"Epoch {ep:3d} | train_loss={avg_train_loss:.6f} "
                f"| train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
            )

    # ---- 最终指标（便于排序/选最优）----
    results["final/train_loss"] = results.get(f"epoch_{n_epochs}/train_loss", None)
    results["final/train_acc"] = results.get(f"epoch_{n_epochs}/train_acc", None)
    results["final/test_acc"] = results.get(f"epoch_{n_epochs}/test_acc", None)

    # ---- 可选：每类准确率（默认不算，避免额外开销）----
    if report_per_class:
        class_correct = [0.0 for _ in range(10)]
        class_total = [0.0 for _ in range(10)]
        with torch.no_grad():
            for data, targets in test_loader:
                x_batch = data.view(-1, 784).T.to(device)
                y_np = encoder.transform(targets.cpu().numpy().reshape(-1, 1))
                y_batch = torch.tensor(y_np, dtype=torch.float32, device=device).T

                acts = model.initialize_activity(x_batch)
                acts = model.update_activity(x_batch, acts, m=m)
                pred_labels = torch.argmax(acts[-1], dim=0)
                true_labels = torch.argmax(y_batch, dim=0)

                c = (pred_labels == true_labels).squeeze()
                for i in range(len(targets)):
                    label = int(true_labels[i].item())
                    class_correct[label] += float(c[i].item())
                    class_total[label] += 1.0

        for i in range(10):
            acc_i = class_correct[i] / max(1.0, class_total[i])
            results[f"class_acc/{i}"] = acc_i

    return results


# --------------------------
# 5. 可选：本地直接跑（不会影响 trainable 被调用）
# --------------------------
if __name__ == "__main__":
    cfg = dict(
        seed=0,
        batch_size=64,
        epochs=50,
        layers=[784, 128, 64, 10],
        m=15,
        lr_activity=0.03,
        lr_weight=0.03,
        lambda_target=0.7,
        use_bias=True,
        device=None,  # None=自动选
        data_root="./data",
        download=True,
        log_every=5,
        report_per_class=False,
    )
    out = trainable(cfg)
    print("final/test_acc =", out["final/test_acc"])
