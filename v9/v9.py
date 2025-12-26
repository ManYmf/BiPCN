def trainable(config):
    """
    Fully-local discPC in a single trainable(config) function.

    Expected config (with defaults):
      - seed: int (0)
      - device: "cuda"|"cpu"|None (auto)
      - input_dim: int (2)
      - hidden_dim: int (64)
      - num_classes: int (2)
      - activation: "tanh"|"relu" ("tanh")
      - batch_size: int (256)
      - epochs: int (20)
      - steps_train: int (20)   # inference steps during training
      - steps_test: int (30)    # inference steps during eval
      - lr_x: float (0.15)      # activity update step
      - lr_w: float (0.02)      # weight update step
      - damp: float (0.1)       # activity damping in [0,1)
      - train_n: int (4096)     # synthetic dataset size
      - test_n: int (1024)
      - class_sep: float (2.0)  # separation of Gaussian blobs
    """
    results = {}

    import math
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # -------------------------
    # 0) config + seed + device
    # -------------------------
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device_cfg = config.get("device", None)
    if device_cfg is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    input_dim = int(config.get("input_dim", 2))
    hidden_dim = int(config.get("hidden_dim", 64))
    num_classes = int(config.get("num_classes", 2))
    activation = str(config.get("activation", "tanh"))

    batch_size = int(config.get("batch_size", 256))
    epochs = int(config.get("epochs", 20))

    steps_train = int(config.get("steps_train", 20))
    steps_test = int(config.get("steps_test", 30))

    lr_x = float(config.get("lr_x", 0.15))
    lr_w = float(config.get("lr_w", 0.02))
    damp = float(config.get("damp", 0.1))

    train_n = int(config.get("train_n", 4096))
    test_n = int(config.get("test_n", 1024))
    class_sep = float(config.get("class_sep", 2.0))

    results["device"] = str(device)
    results["torch.cuda.is_available()"] = bool(torch.cuda.is_available())

    # -------------------------
    # 1) local helpers (inside)
    # -------------------------
    def _act(x):
        if activation == "tanh":
            return torch.tanh(x)
        if activation == "relu":
            return torch.relu(x)
        raise ValueError(f"Unsupported activation: {activation}")

    def _act_prime(x):
        if activation == "tanh":
            y = torch.tanh(x)
            return 1.0 - y * y
        if activation == "relu":
            return (x > 0).to(x.dtype)
        raise ValueError(f"Unsupported activation: {activation}")

    def _one_hot(y):
        return torch.nn.functional.one_hot(y, num_classes=num_classes).float()

    def _make_blobs(n):
        # Two Gaussian blobs (generalized to num_classes by cycling means on a circle if needed)
        # For simplicity and stability, keep num_classes=2 as default.
        g = torch.Generator(device="cpu")
        g.manual_seed(seed + n)

        X = torch.randn(n, input_dim, generator=g)
        y = torch.randint(0, num_classes, (n,), generator=g)

        # place class means
        if num_classes == 2:
            m0 = torch.zeros(input_dim)
            m1 = torch.zeros(input_dim)
            m0[0] = -class_sep / 2.0
            m1[0] = +class_sep / 2.0
            means = torch.stack([m0, m1], dim=0)  # [2, D]
        else:
            # arrange means on a circle in first 2 dims
            means = torch.zeros(num_classes, input_dim)
            for k in range(num_classes):
                ang = 2.0 * math.pi * k / float(num_classes)
                means[k, 0] = math.cos(ang) * (class_sep / 2.0)
                if input_dim > 1:
                    means[k, 1] = math.sin(ang) * (class_sep / 2.0)

        X = X + means[y]
        return X.to(device), y.to(device)

    # -------------------------
    # 2) discPC model (fully local)
    # -------------------------
    layer_sizes = [input_dim, hidden_dim, num_classes]
    L = len(layer_sizes) - 1  # transitions count

    # V[l] : maps x[l] -> x[l+1] predictor, shape [d_{l+1}, d_l]
    V = []
    gW = torch.Generator(device="cpu")
    gW.manual_seed(seed + 123)
    for l in range(L):
        W = 0.01 * torch.randn(layer_sizes[l + 1], layer_sizes[l], generator=gW)
        V.append(W.to(device))

    def _ff_init(x0):
        xs = [x0]
        for l in range(1, L + 1):
            pred = (V[l - 1] @ _act(xs[l - 1]).T).T  # [B, d_l]
            xs.append(pred.contiguous())
        return xs

    def _compute_eps(xs):
        # eps[0] unused placeholder
        eps = [torch.zeros_like(xs[0])]
        for l in range(1, L + 1):
            pred = (V[l - 1] @ _act(xs[l - 1]).T).T
            eps.append(xs[l] - pred)
        return eps

    def _infer(xs, clamp, T, lr_x_local):
        # local activity updates, no autograd
        for _ in range(T):
            eps = _compute_eps(xs)

            # hidden layers: l=1..L-1
            for l in range(1, L):
                if clamp[l]:
                    continue
                back = (V[l].T @ eps[l + 1].T).T  # [B, d_l]
                grad = eps[l] - back * _act_prime(xs[l])
                x_new = xs[l] - lr_x_local * grad
                xs[l] = (1.0 - damp) * x_new + damp * xs[l]

            # output layer: grad = eps[L] if not clamped
            if not clamp[L]:
                gradL = eps[L]
                x_new = xs[L] - lr_x_local * gradL
                xs[L] = (1.0 - damp) * x_new + damp * xs[L]

        return xs

    def _update_weights(xs, lr_w_local):
        # Hebbian local weight update: ΔV[l-1] = lr_w * (eps[l]^T @ f(x[l-1])) / B
        eps = _compute_eps(xs)
        B = xs[0].shape[0]
        invB = 1.0 / float(B)
        for l in range(1, L + 1):
            pre = _act(xs[l - 1])        # [B, d_{l-1}]
            post_err = eps[l]            # [B, d_l]
            dV = (post_err.T @ pre) * invB  # [d_l, d_{l-1}]
            V[l - 1].add_(lr_w_local * dV)

    def _predict_logits(x0, T):
        xs = _ff_init(x0)
        clamp = [True] + [False] * L  # clamp input only
        xs = _infer(xs, clamp, T=T, lr_x_local=lr_x)
        return xs[L]

    def _accuracy(loader, T):
        correct = 0
        total = 0
        for xb, yb in loader:
            logits = _predict_logits(xb, T=T)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
        return correct / max(total, 1)

    # -------------------------
    # 3) data (synthetic, no external deps)
    # -------------------------
    Xtr, ytr = _make_blobs(train_n)
    Xte, yte = _make_blobs(test_n)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)

    # -------------------------
    # 4) train loop (local PC)
    # -------------------------
    for ep in range(1, epochs + 1):
        for xb, yb in train_loader:
            # init activities
            xs = _ff_init(xb)

            # supervised clamp: input & output clamped, hidden free
            xs[0] = xb
            xs[L] = _one_hot(yb)
            clamp = [True] + [False] * (L - 1) + [True]

            xs = _infer(xs, clamp, T=steps_train, lr_x_local=lr_x)
            _update_weights(xs, lr_w_local=lr_w)

        acc = _accuracy(test_loader, T=steps_test)
        results[f"epoch_{ep}/test_acc"] = float(acc)

    # return final weights norms as sanity metrics
    with torch.no_grad():
        for l in range(L):
            results[f"V{l}_fro"] = float(torch.norm(V[l]).item())

    return results
