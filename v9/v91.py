def trainable(config):
    """
    discPC (fully local updates) on MNIST, single trainable(config).

    Key defaults aligned with the paper's MNIST MLP setting:
      - layer_sizes: [784, 256, 256, 10]  (MLP for MNIST/Fashion-MNIST) :contentReference[oaicite:2]{index=2}
      - input normalized to [-1, 1] :contentReference[oaicite:3]{index=3}
      - eval inference steps ~100 :contentReference[oaicite:4]{index=4}
    """
    results = {}

    import math
    import torch
    from torch.utils.data import DataLoader

    # -------------------------
    # 0) config + device + seed
    # -------------------------
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device_cfg = config.get("device", None)
    device = torch.device(device_cfg) if device_cfg is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    results["device"] = str(device)
    results["torch.cuda.is_available()"] = bool(torch.cuda.is_available())

    # Model size (default: 784->256->256->10) :contentReference[oaicite:5]{index=5}
    hidden1 = int(config.get("hidden1", 256))
    hidden2 = int(config.get("hidden2", 256))
    layer_sizes = [784, hidden1, hidden2, 10]
    L = len(layer_sizes) - 1  # number of prediction links

    # PC hyperparams
    batch_size = int(config.get("batch_size", 256))
    epochs = int(config.get("epochs", 10))

    steps_train = int(config.get("steps_train", 20))
    steps_test = int(config.get("steps_test", 100))  # common eval choice :contentReference[oaicite:6]{index=6}

    lr_x = float(config.get("lr_x", 0.05))
    lr_w = float(config.get("lr_w", 0.01))
    damp = float(config.get("damp", 0.1))  # activity damping in [0,1)

    # activation used for hidden-to-hidden predictions; last (to logits) uses identity :contentReference[oaicite:7]{index=7}
    activation = str(config.get("activation", "tanh"))  # "tanh"|"relu"|"leaky_relu"
    leaky_slope = float(config.get("leaky_slope", 0.01))

    use_bias = bool(config.get("use_bias", True))
    weight_decay = float(config.get("weight_decay", 0.0))  # optional local decay

    # log config into results (like your tables)
    results["config/batch_size"] = batch_size
    results["config/epochs"] = epochs
    results["config/steps_train"] = steps_train
    results["config/steps_test"] = steps_test
    results["config/lr_x"] = lr_x
    results["config/lr_w"] = lr_w
    results["config/damp"] = damp
    results["config/hidden1"] = hidden1
    results["config/hidden2"] = hidden2
    results["config/activation"] = activation
    results["config/use_bias"] = use_bias
    results["config/weight_decay"] = weight_decay

    # -------------------------
    # 1) MNIST loaders (inside)
    # -------------------------
    from torchvision import datasets, transforms

    # normalize to [-1, 1] as described in the supplement :contentReference[oaicite:8]{index=8}
    tfm = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # -------------------------
    # 2) Local helpers (inside)
    # -------------------------
    def _one_hot(y):
        return torch.nn.functional.one_hot(y, num_classes=10).float()

    def _hid_act(x):
        if activation == "tanh":
            return torch.tanh(x)
        if activation == "relu":
            return torch.relu(x)
        if activation == "leaky_relu":
            return torch.where(x > 0, x, leaky_slope * x)
        raise ValueError(f"Unsupported activation: {activation}")

    def _hid_act_prime(x):
        if activation == "tanh":
            y = torch.tanh(x)
            return 1.0 - y * y
        if activation == "relu":
            return (x > 0).to(x.dtype)
        if activation == "leaky_relu":
            return torch.where(x > 0, torch.ones_like(x), torch.full_like(x, leaky_slope))
        raise ValueError(f"Unsupported activation: {activation}")

    # activation used on the PRE-synaptic side of each predictor V[i]: layer i -> i+1
    # per paper: disc prediction to x_L often uses identity (no activation) :contentReference[oaicite:9]{index=9}
    def _pre_act(i, x_i):
        # i is the source layer index (0..L-1). If predicting into output layer (i+1==L), use identity
        if (i + 1) == L:
            return x_i
        return _hid_act(x_i)

    # derivative for backprop term at layer l w.r.t. using layer l as pre-synaptic input to V[l]
    def _pre_act_prime(l, x_l):
        # l is the layer index (1..L-1) when used as input to V[l] (predicting l+1)
        if (l + 1) == L:
            return torch.ones_like(x_l)  # identity
        return _hid_act_prime(x_l)

    # -------------------------
    # 3) Parameters: V and bias
    # -------------------------
    gW = torch.Generator(device="cpu")
    gW.manual_seed(seed + 123)

    V = []
    b = []  # bias for each predictor V[i] (optional), shape [d_{i+1}]
    for i in range(L):
        W = 0.01 * torch.randn(layer_sizes[i + 1], layer_sizes[i], generator=gW)
        V.append(W.to(device))
        if use_bias:
            bb = torch.zeros(layer_sizes[i + 1])
            b.append(bb.to(device))
        else:
            b.append(None)

    # -------------------------
    # 4) discPC routines (local)
    # -------------------------
    @torch.no_grad()
    def _ff_init(x0):
        xs = [x0]
        for i in range(L):
            pre = _pre_act(i, xs[i])
            pred = (V[i] @ pre.T).T
            if b[i] is not None:
                pred = pred + b[i].view(1, -1)
            xs.append(pred.contiguous())
        return xs  # length L+1

    @torch.no_grad()
    def _compute_eps(xs):
        # eps[0] unused placeholder; eps[l] corresponds to x[l] - pred from below (l=1..L)
        eps = [torch.zeros_like(xs[0])]
        for l in range(1, L + 1):
            i = l - 1  # predictor index
            pre = _pre_act(i, xs[i])
            pred = (V[i] @ pre.T).T
            if b[i] is not None:
                pred = pred + b[i].view(1, -1)
            eps.append(xs[l] - pred)
        return eps

    @torch.no_grad()
    def _infer(xs, clamp, T):
        # clamp: list[bool] length L+1
        for _ in range(T):
            eps = _compute_eps(xs)

            # update hidden layers l=1..L-1
            for l in range(1, L):
                if clamp[l]:
                    continue
                back = (V[l].T @ eps[l + 1].T).T
                grad = eps[l] - back * _pre_act_prime(l, xs[l])
                x_new = xs[l] - lr_x * grad
                xs[l] = (1.0 - damp) * x_new + damp * xs[l]

            # update output layer (if free): grad = eps[L]
            if not clamp[L]:
                x_new = xs[L] - lr_x * eps[L]
                xs[L] = (1.0 - damp) * x_new + damp * xs[L]

        return xs

    @torch.no_grad()
    def _update_weights(xs):
        eps = _compute_eps(xs)
        B = xs[0].shape[0]
        invB = 1.0 / float(B)

        for l in range(1, L + 1):
            i = l - 1
            pre = _pre_act(i, xs[i])              # [B, d_i]
            post_err = eps[l]                     # [B, d_{i+1}]
            dV = (post_err.T @ pre) * invB        # [d_{i+1}, d_i]
            V[i].add_(lr_w * dV)

            if b[i] is not None:
                db = post_err.mean(dim=0)         # [d_{i+1}]
                b[i].add_(lr_w * db)

            if weight_decay > 0.0:
                V[i].mul_(1.0 - lr_w * weight_decay)

    @torch.no_grad()
    def _predict_logits(x0, T):
        xs = _ff_init(x0)
        clamp = [True] + [False] * L  # clamp input only
        xs = _infer(xs, clamp, T=T)
        return xs[L]

    @torch.no_grad()
    def _accuracy(loader, T):
        correct = 0
        total = 0
        for x_img, y in loader:
            x = x_img.to(device).view(x_img.size(0), -1)
            x = x * 2.0 - 1.0  # [-1,1] :contentReference[oaicite:10]{index=10}
            y = y.to(device)
            logits = _predict_logits(x, T=T)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(total, 1)

    # -------------------------
    # 5) Train loop
    # -------------------------
    for ep in range(1, epochs + 1):
        for x_img, y in train_loader:
            x = x_img.to(device).view(x_img.size(0), -1)
            x = x * 2.0 - 1.0  # [-1,1] :contentReference[oaicite:11]{index=11}
            y = y.to(device)
            y_oh = _one_hot(y)

            xs = _ff_init(x)

            # supervised clamp: input and label clamped, hidden free
            xs[0] = x
            xs[L] = y_oh
            clamp = [True] + [False] * (L - 1) + [True]

            xs = _infer(xs, clamp, T=steps_train)
            _update_weights(xs)

        acc = _accuracy(test_loader, T=steps_test)
        results[f"epoch_{ep}/test_acc"] = float(acc)

    # some norms for debugging/monitoring
    with torch.no_grad():
        for i in range(L):
            results[f"V{i}_fro"] = float(torch.norm(V[i]).item())
            if b[i] is not None:
                results[f"b{i}_l2"] = float(torch.norm(b[i]).item())

    return results
