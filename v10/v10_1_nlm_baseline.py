# --------------------------
# Baselines:
#   1) backprop: train all layers
#   2) nonlinear_mapping: freeze prefix (random nonlinear feature map), train last linear(s)
# --------------------------
import torch.nn as nn
import numpy as np
import torch
from v10_1 import load_mnist_data


def _make_act_module(name: str, negative_slope: float = 0.01) -> nn.Module:
    name = (name or "leaky_relu").lower()
    if name in ["identity", "linear", "none"]:
        return nn.Identity()
    if name == "tanh":
        return nn.Tanh()
    if name in ["leaky_relu", "lrelu"]:
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class BP_MLP(nn.Module):
    """
    BP 版 MLP（支持 Nonlinear Mapping：冻结前面层，仅训练最后若干个 Linear）
    - 输入: (B, 784)
    - 输出: (B, 10) logits
    """

    def __init__(
        self,
        layers,
        activation="leaky_relu",
        use_bias=True,
        negative_slope=0.01,
        weight_init="xavier",
    ):
        super().__init__()
        layers = list(layers)
        assert len(layers) >= 2, "layers must have at least [in, out]"

        act = _make_act_module(activation, negative_slope=negative_slope)

        mods = []
        self.linears = nn.ModuleList()

        for i in range(len(layers) - 1):
            in_d, out_d = int(layers[i]), int(layers[i + 1])
            lin = nn.Linear(in_d, out_d, bias=bool(use_bias))

            # weight init（尽量对齐你原文件的 xavier / normal 逻辑）
            if (weight_init or "xavier").lower() == "xavier":
                nn.init.xavier_uniform_(lin.weight)
            else:
                nn.init.normal_(lin.weight, mean=0.0, std=(1.0 / max(1, in_d)) ** 0.5)

            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

            self.linears.append(lin)
            mods.append(lin)

            # 最后一层不加激活（输出 logits）
            if i < len(layers) - 2:
                mods.append(act)

        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)

    def set_trainable_last_n_linear(self, n_last: int = 1):
        """
        冻结前面的 Linear，仅训练最后 n_last 个 Linear。
        注意：激活层无参数，无需处理。
        """
        n_last = int(n_last)
        if n_last <= 0:
            raise ValueError("n_last must be >= 1")

        n_total = len(self.linears)
        if n_last > n_total:
            raise ValueError(f"n_last ({n_last}) > #linears ({n_total})")

        # 先全部冻结
        for p in self.parameters():
            p.requires_grad = False

        # 解冻最后 n_last 个 Linear
        for lin in list(self.linears)[n_total - n_last :]:
            for p in lin.parameters():
                p.requires_grad = True


def trainable_bp_or_nonlinear_mapping(config: dict):
    """
    两种模式共用一个 trainable，靠 config["mode"] 控制：
      - mode="backprop"           -> 训练全部参数（标准 BP baseline）
      - mode="nonlinear_mapping"  -> 冻结前面层，仅训练最后 n 个 Linear（默认 n=1）

    其余接口/日志字段尽量贴合你现有 trainable(config) 风格。
    """
    results = {}

    # ---- seed / device ----
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_cfg = config.get("device", None)
    device = (
        torch.device(device_cfg)
        if device_cfg is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    results["device"] = str(device)
    results["torch.cuda.is_available"] = bool(torch.cuda.is_available())

    # ---- mode ----
    mode = (config.get("mode", "backprop") or "backprop").lower()
    if mode not in ["backprop", "nonlinear_mapping"]:
        raise ValueError(f"Unknown mode: {mode}")
    results["config/mode"] = mode

    # ---- data (reuse your loader) ----
    batch_size = int(config.get("batch_size", 256))
    data_root = config.get("data_root", None)
    download = bool(config.get("download", False))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", False))
    user_name = config.get("user_name", "mengfan")

    train_loader, test_loader = load_mnist_data(
        batch_size=batch_size,
        data_root=data_root,
        download=download,
        num_workers=num_workers,
        pin_memory=pin_memory,
        user_name=user_name,
    )

    # ---- model config ----
    layers = config.get("layers", [784, 256, 256, 10])
    activation = config.get("activation", "leaky_relu")
    negative_slope = float(config.get("negative_slope", 0.01))
    use_bias = bool(config.get("use_bias", True))
    weight_init = config.get("weight_init", "xavier")

    # ---- optim config ----
    lr_theta = float(config.get("lr_theta", 1e-3))
    weight_decay = float(config.get("weight_decay", 5e-4))

    optimizer_name = (config.get("optimizer", "sgd") or "sgd").lower()
    sgd_momentum = float(config.get("sgd_momentum", 0.9))
    sgd_nesterov = bool(config.get("sgd_nesterov", False))

    adam_beta1 = float(config.get("adam_beta1", 0.9))
    adam_beta2 = float(config.get("adam_beta2", 0.999))
    adam_eps = float(config.get("adam_eps", 1e-8))

    n_epochs = int(config.get("epochs", 25))
    log_every = int(config.get("log_every", 5))

    # ---- record config fields (for dataframe) ----
    results["config/method"] = "backprop" if mode == "backprop" else "nonlinear_mapping"
    results["config/seed"] = seed
    results["config/batch_size"] = batch_size
    results["config/epochs"] = n_epochs
    results["config/layers"] = str(layers)
    results["config/activation"] = str(activation)
    results["config/negative_slope"] = negative_slope
    results["config/use_bias"] = use_bias
    results["config/weight_init"] = str(weight_init)

    results["config/lr_theta"] = lr_theta
    results["config/weight_decay"] = weight_decay

    results["config/optimizer"] = optimizer_name
    results["config/sgd_momentum"] = sgd_momentum
    results["config/sgd_nesterov"] = sgd_nesterov
    results["config/adam_beta1"] = adam_beta1
    results["config/adam_beta2"] = adam_beta2
    results["config/adam_eps"] = adam_eps

    # ---- build model ----
    model = BP_MLP(
        layers=layers,
        activation=activation,
        use_bias=use_bias,
        negative_slope=negative_slope,
        weight_init=weight_init,
    ).to(device)

    # ---- Nonlinear Mapping: freeze prefix ----
    train_last_n_linear = int(config.get("train_last_n_linear", 1))
    results["config/train_last_n_linear"] = train_last_n_linear

    if mode == "nonlinear_mapping":
        model.set_trainable_last_n_linear(train_last_n_linear)

    # ---- optimizer (only params that require grad) ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. Check train_last_n_linear / freezing logic.")

    if optimizer_name in ["sgd"]:
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=lr_theta,
            momentum=sgd_momentum,
            weight_decay=weight_decay,
            nesterov=sgd_nesterov,
        )
    elif optimizer_name in ["adamw", "adam"]:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr_theta,
            betas=(adam_beta1, adam_beta2),
            eps=adam_eps,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    criterion = nn.CrossEntropyLoss()

    # ---- train / eval ----
    for epoch in range(n_epochs):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for data, targets in train_loader:
            x = data.view(data.size(0), -1).to(device=device, dtype=torch.float32)
            y = targets.to(device=device, dtype=torch.long)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bsz = x.size(0)
            train_loss_sum += float(loss.item()) * bsz
            train_n += bsz

        train_loss = train_loss_sum / max(1, train_n)

        model.eval()
        test_correct = 0
        test_total = 0
        test_loss_sum = 0.0

        with torch.no_grad():
            for data, targets in test_loader:
                x = data.view(data.size(0), -1).to(device=device, dtype=torch.float32)
                y = targets.to(device=device, dtype=torch.long)

                logits = model(x)
                loss = criterion(logits, y)

                pred = torch.argmax(logits, dim=1)
                test_correct += int((pred == y).sum().item())
                test_total += int(y.numel())
                test_loss_sum += float(loss.item()) * x.size(0)

        test_acc = test_correct / max(1, test_total)
        test_loss = test_loss_sum / max(1, test_total)

        ep = epoch + 1
        results[f"epoch_{ep}/train_loss"] = float(train_loss)
        results[f"epoch_{ep}/test_loss"] = float(test_loss)
        results[f"epoch_{ep}/test_acc"] = float(test_acc)

        if log_every > 0 and (ep % log_every == 0 or ep == 1 or ep == n_epochs):
            tag = "BP" if mode == "backprop" else f"NM(last{train_last_n_linear})"
            print(
                f"[{tag}:{optimizer_name.upper()}] Epoch {ep:3d} | "
                f"train_loss={train_loss:.4f} | test_acc={test_acc:.4f} | test_loss={test_loss:.4f}"
            )

    results["final/test_acc"] = results.get(f"epoch_{n_epochs}/test_acc", None)
    results["final/train_loss"] = results.get(f"epoch_{n_epochs}/train_loss", None)
    results["final/test_loss"] = results.get(f"epoch_{n_epochs}/test_loss", None)
    return results


# 兼容你之前的命名：如果你外部脚本在 import trainable_backprop，可以保留这个别名
def trainable_backprop(config: dict):
    config = dict(config)
    config.setdefault("mode", "backprop")
    return trainable_bp_or_nonlinear_mapping(config)


def trainable_nonlinear_mapping(config: dict):
    config = dict(config)
    config.setdefault("mode", "nonlinear_mapping")
    config.setdefault("train_last_n_linear", 1)
    return trainable_bp_or_nonlinear_mapping(config)


if __name__ == "__main__":
    # ---- 1) Backprop sanity run ----
    cfg_bp = dict(
        seed=0,
        device=None,
        data_root=None,
        download=False,
        user_name="mengfan",
        batch_size=256,
        epochs=5,
        layers=[784, 256, 256, 10],
        activation="leaky_relu",
        negative_slope=0.01,
        use_bias=True,
        weight_init="xavier",
        mode="backprop",
        optimizer="sgd",
        sgd_momentum=0.9,
        sgd_nesterov=False,
        lr_theta=0.05,
        weight_decay=5e-4,
        log_every=1,
    )
    out_bp = trainable_bp_or_nonlinear_mapping(cfg_bp)
    print("BP final/test_acc =", out_bp["final/test_acc"])

    # ---- 2) Nonlinear Mapping sanity run (freeze all but last linear) ----
    cfg_nm = dict(
        seed=0,
        device=None,
        data_root=None,
        download=False,
        user_name="mengfan",
        batch_size=256,
        epochs=5,
        layers=[784, 256, 256, 10],
        activation="leaky_relu",
        negative_slope=0.01,
        use_bias=True,
        weight_init="xavier",
        mode="nonlinear_mapping",
        train_last_n_linear=1,     # 只训练最后一层（读出层）
        optimizer="sgd",
        sgd_momentum=0.9,
        sgd_nesterov=False,
        lr_theta=0.1,              # 只训最后一层时，lr 往往可更大
        weight_decay=5e-4,
        log_every=1,
    )
    out_nm = trainable_bp_or_nonlinear_mapping(cfg_nm)
    print("NM final/test_acc =", out_nm["final/test_acc"])
