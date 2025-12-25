# --------------------------
# Backprop baseline (MLP + CrossEntropy)
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
    与你当前 BiPCN_MLP 的 layers/activation/use_bias/weight_init 对齐的 BP 版 MLP。
    - 输入: (B, 784)
    - 输出: (B, 10) logits
    """
    def __init__(self, layers, activation="leaky_relu", use_bias=True,
                 negative_slope=0.01, weight_init="xavier"):
        super().__init__()
        layers = list(layers)
        assert len(layers) >= 2, "layers must have at least [in, out]"

        act = _make_act_module(activation, negative_slope=negative_slope)

        mods = []
        for i in range(len(layers) - 1):
            in_d, out_d = int(layers[i]), int(layers[i + 1])
            lin = nn.Linear(in_d, out_d, bias=bool(use_bias))

            # weight init（尽量对齐你文件里的 xavier / normal 逻辑）
            if (weight_init or "xavier").lower() == "xavier":
                nn.init.xavier_uniform_(lin.weight)
            else:
                # std ~ sqrt(1/in_d)（和你当前文件中的“非xavier”分支一致风格）
                nn.init.normal_(lin.weight, mean=0.0, std=(1.0 / max(1, in_d)) ** 0.5)

            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

            mods.append(lin)
            # 最后一层不加激活（输出 logits）
            if i < len(layers) - 2:
                mods.append(act)

        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def trainable_backprop(config: dict):
    """
    Backprop baseline，接口/日志字段尽量贴合你当前的 trainable(config)：
    - 复用 load_mnist_data
    - 复用 epochs/batch_size/lr_theta/weight_decay/adam_beta1/adam_beta2/adam_eps 等 key
    - 输出 epoch_{k}/test_acc, final/test_acc 等
    """
    results = {}

    # ---- seed / device ----
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_cfg = config.get("device", None)
    device = torch.device(device_cfg) if device_cfg is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    results["device"] = str(device)
    results["torch.cuda.is_available"] = bool(torch.cuda.is_available())

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

    # ---- optim config (reuse your theta keys) ----
    lr_theta = float(config.get("lr_theta", 1e-4))
    weight_decay = float(config.get("weight_decay", 5e-3))
    adam_beta1 = float(config.get("adam_beta1", 0.9))
    adam_beta2 = float(config.get("adam_beta2", 0.999))
    adam_eps = float(config.get("adam_eps", 1e-8))

    n_epochs = int(config.get("epochs", 25))
    log_every = int(config.get("log_every", 5))

    # ---- record config fields (for dataframe) ----
    results["config/method"] = "backprop"
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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_theta,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
        weight_decay=weight_decay,
    )

    # ---- train / eval ----
    for epoch in range(n_epochs):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for data, targets in train_loader:
            x = data.view(data.size(0), -1).to(device=device, dtype=torch.float32)  # (B,784)
            y = targets.to(device=device, dtype=torch.long)

            logits = model(x)  # (B,10)
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
            print(f"[BP] Epoch {ep:3d} | train_loss={train_loss:.4f} | test_acc={test_acc:.4f} | test_loss={test_loss:.4f}")

    results["final/test_acc"] = results.get(f"epoch_{n_epochs}/test_acc", None)
    results["final/train_loss"] = results.get(f"epoch_{n_epochs}/train_loss", None)
    results["final/test_loss"] = results.get(f"epoch_{n_epochs}/test_loss", None)
    return results


if __name__ == "__main__":
    # quick sanity run
    cfg = dict(
        seed=0,
        device=None,
        data_root=None,     # None -> radas.get_data_dir(user)/data
        download=False,
        user_name="mengfan",
        batch_size=256,
        epochs=10,
        layers=[784, 256, 256, 10],
        activation="leaky_relu",
        negative_slope=0.01,
        use_bias=True,
        weight_init="xavier",
        lr_theta=1e-3,        # BP 通常可以更大一些；你也可以和 bPC 保持一致
        weight_decay=5e-4,
        log_every=1,
    )
    out = trainable_backprop(cfg)
    print("final/test_acc =", out["final/test_acc"])
