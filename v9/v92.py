# 完全按照论文中的实现方式，实现discPC

def trainable(config):
    """
    discPC implementation aligned to Oliviers et al. (2025) "Bidirectional predictive coding"
    by taking the bPC dynamics (Eq. 5-7) and setting alpha_gen = 0, alpha_disc > 0.
    - Energy (discPC): E_disc = sum_{l=2..L} alpha_disc/2 * || x_l - (V_{l-1} f(x_{l-1}) + b_{l-1}) ||^2
    - Errors: eps_l = alpha_disc * (x_l - pred_l)  (Eq. 6, disc part)
    - Neural dynamics: dx_l/dt = -eps_l + f'(x_l) ⊙ (V_l^T eps_{l+1}) (Eq. 5 specialized)
    - Weight updates: grad(V_l) = - eps_{l+1} f(x_l)^T, grad(b_l) = - mean(eps_{l+1})
      then parameter optimizer step (SGD/AdamW) (Eq. 7 specialized)
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_cfg = config.get("device", None)
    device = torch.device("cuda" if (device_cfg is None and torch.cuda.is_available()) else (device_cfg or "cpu"))

    # Model sizes: allow either hidden_dim or hidden_dims list
    input_dim = int(config.get("input_dim", 2))
    num_classes = int(config.get("num_classes", 2))
    if "hidden_dims" in config and config["hidden_dims"] is not None:
        hidden_dims = [int(x) for x in config["hidden_dims"]]
    else:
        hidden_dims = [int(config.get("hidden_dim", 64))]

    layer_sizes = [input_dim] + hidden_dims + [num_classes]
    L = len(layer_sizes) - 1  # number of transitions

    # Activation choices mentioned in the paper's search space
    activation = str(config.get("activation", "tanh")).lower()
    leaky_slope = float(config.get("leaky_slope", 0.01))

    # Training setup (paper often uses batch=256, epochs=25, T=8, T_eval=100 for MLP)
    batch_size = int(config.get("batch_size", 256))
    epochs = int(config.get("epochs", 25))
    steps_train = int(config.get("steps_train", 8))
    steps_test = int(config.get("steps_test", 100))

    # Disc energy scaling (alpha_disc); discPC => alpha_gen = 0
    alpha_disc = float(config.get("alpha_disc", 1.0))

    # Activity optimiser (SGD w/ momentum), matches "momentum_x" knob in paper tables
    # Keep backward compatibility with your lr_x/damp keys:
    lr_x = float(config.get("lrx", config.get("lr_x", 0.05)))
    momentum_x = float(config.get("momentum_x", 0.0))

    # Parameter optimiser settings (lrθ + weight_decayθ). Provide AdamW by default.
    lr_theta = float(config.get("lr_theta", config.get("lr_w", 3e-4)))
    weight_decay = float(config.get("weight_decay", 0.0))
    param_optim = str(config.get("param_optim", "adamw")).lower()

    # AdamW hyperparams
    adam_beta1 = float(config.get("adam_beta1", 0.9))
    adam_beta2 = float(config.get("adam_beta2", 0.999))
    adam_eps = float(config.get("adam_eps", 1e-8))

    # Synthetic data
    train_n = int(config.get("train_n", 4096))
    test_n = int(config.get("test_n", 1024))
    class_sep = float(config.get("class_sep", 2.0))

    results["device"] = str(device)
    results["torch.cuda.is_available()"] = bool(torch.cuda.is_available())

    # -------------------------
    # 1) local helpers
    # -------------------------
    def _act(x):
        if activation == "tanh":
            return torch.tanh(x)
        if activation == "relu":
            return torch.relu(x)
        if activation in ("leaky_relu", "leaky-relu", "lrelu"):
            return torch.where(x >= 0, x, leaky_slope * x)
        if activation == "gelu":
            # tanh approximation (same family as used in many implementations)
            return 0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
            ))
        raise ValueError(f"Unsupported activation: {activation}")

    def _act_prime(x):
        if activation == "tanh":
            y = torch.tanh(x)
            return 1.0 - y * y
        if activation == "relu":
            return (x > 0).to(x.dtype)
        if activation in ("leaky_relu", "leaky-relu", "lrelu"):
            return torch.where(x >= 0, torch.ones_like(x), torch.full_like(x, leaky_slope))
        if activation == "gelu":
            # derivative of tanh-approx GELU
            k = math.sqrt(2.0 / math.pi)
            a = 0.044715
            x2 = x * x
            x3 = x2 * x
            u = k * (x + a * x3)
            t = torch.tanh(u)
            sech2 = 1.0 - t * t
            du_dx = k * (1.0 + 3.0 * a * x2)
            return 0.5 * (1.0 + t) + 0.5 * x * sech2 * du_dx
        raise ValueError(f"Unsupported activation: {activation}")

    def _one_hot(y):
        return torch.nn.functional.one_hot(y, num_classes=num_classes).float()

    def _make_blobs(n):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed + n)

        X = torch.randn(n, input_dim, generator=g)
        y = torch.randint(0, num_classes, (n,), generator=g)

        if num_classes == 2:
            m0 = torch.zeros(input_dim)
            m1 = torch.zeros(input_dim)
            m0[0] = -class_sep / 2.0
            m1[0] = +class_sep / 2.0
            means = torch.stack([m0, m1], dim=0)
        else:
            means = torch.zeros(num_classes, input_dim)
            for k in range(num_classes):
                ang = 2.0 * math.pi * k / float(num_classes)
                means[k, 0] = math.cos(ang) * (class_sep / 2.0)
                if input_dim > 1:
                    means[k, 1] = math.sin(ang) * (class_sep / 2.0)

        X = X + means[y]
        return X.to(device), y.to(device)

    # -------------------------
    # 2) Parameters: V and bias b (bottom-up only)
    # -------------------------
    gW = torch.Generator(device="cpu")
    gW.manual_seed(seed + 123)

    V = []
    b = []
    for l in range(L):
        # small normal init (stable for local dynamics)
        W = 0.01 * torch.randn(layer_sizes[l + 1], layer_sizes[l], generator=gW)
        bb = torch.zeros(layer_sizes[l + 1])
        V.append(W.to(device))
        b.append(bb.to(device))

    # Parameter optimiser state (manual, no autograd)
    t_adam = 0
    mV = [torch.zeros_like(Vl) for Vl in V]
    vV = [torch.zeros_like(Vl) for Vl in V]
    mb = [torch.zeros_like(bl) for bl in b]
    vb = [torch.zeros_like(bl) for bl in b]

    def _ff_init(x0):
        """Bottom-up feedforward sweep initialization (paper protocol)."""
        xs = [x0]
        for l in range(1, L + 1):
            pred = (V[l - 1] @ _act(xs[l - 1]).T).T + b[l - 1]
            xs.append(pred.contiguous())
        return xs

    def _compute_eps_disc(xs):
        """eps_disc[l] = alpha_disc * (x_l - pred_l). eps_disc[0] unused."""
        eps = [torch.zeros_like(xs[0])]
        for l in range(1, L + 1):
            pred = (V[l - 1] @ _act(xs[l - 1]).T).T + b[l - 1]
            eps.append(alpha_disc * (xs[l] - pred))
        return eps

    def _infer_disc(xs, clamp, T):
        """
        Discretized disc-only neural dynamics from Eq. (5):
          dx_l/dt = -eps_l + f'(x_l) ⊙ (V_l^T eps_{l+1})
        implemented as SGD on energy with momentum_x.
        """
        # velocities for momentum SGD on activities (only for unclamped layers)
        vxs = [torch.zeros_like(x) for x in xs]

        for _ in range(T):
            eps = _compute_eps_disc(xs)

            # update hidden layers l=1..L-1
            for l in range(1, L):
                if clamp[l]:
                    continue
                back = (V[l].T @ eps[l + 1].T).T  # [B, d_l]
                grad = eps[l] - back * _act_prime(xs[l])  # gradient of energy w.r.t x_l
                # momentum SGD: v = mu*v - lr*grad ; x += v
                vxs[l].mul_(momentum_x).add_(-lr_x * grad)
                xs[l].add_(vxs[l])

            # output layer l=L
            if not clamp[L]:
                gradL = eps[L]
                vxs[L].mul_(momentum_x).add_(-lr_x * gradL)
                xs[L].add_(vxs[L])

        return xs

    def _param_step(gradsV, gradsb):
        """Apply either SGD(+weight_decay) or AdamW(+decoupled weight decay) to parameters."""
        nonlocal t_adam
        if param_optim == "sgd":
            for l in range(L):
                if weight_decay != 0.0:
                    V[l].add_(-lr_theta * weight_decay * V[l])
                    b[l].add_(-lr_theta * weight_decay * b[l])
                V[l].add_(-lr_theta * gradsV[l])
                b[l].add_(-lr_theta * gradsb[l])
            return

        if param_optim != "adamw":
            raise ValueError(f"param_optim must be 'adamw' or 'sgd', got {param_optim}")

        t_adam += 1
        bias_c1 = 1.0 - (adam_beta1 ** t_adam)
        bias_c2 = 1.0 - (adam_beta2 ** t_adam)

        for l in range(L):
            gV = gradsV[l]
            gb_ = gradsb[l]

            # Decoupled weight decay (AdamW)
            if weight_decay != 0.0:
                V[l].add_(-lr_theta * weight_decay * V[l])
                b[l].add_(-lr_theta * weight_decay * b[l])

            # Adam moments
            mV[l].mul_(adam_beta1).add_((1.0 - adam_beta1) * gV)
            vV[l].mul_(adam_beta2).add_((1.0 - adam_beta2) * (gV * gV))
            mb[l].mul_(adam_beta1).add_((1.0 - adam_beta1) * gb_)
            vb[l].mul_(adam_beta2).add_((1.0 - adam_beta2) * (gb_ * gb_))

            mV_hat = mV[l] / bias_c1
            vV_hat = vV[l] / bias_c2
            mb_hat = mb[l] / bias_c1
            vb_hat = vb[l] / bias_c2

            V[l].add_(-lr_theta * mV_hat / (torch.sqrt(vV_hat) + adam_eps))
            b[l].add_(-lr_theta * mb_hat / (torch.sqrt(vb_hat) + adam_eps))

    def _update_params_disc(xs):
        """
        Eq. (7) specialized to discPC:
          grad(V_l) = - eps_{l+1} f(x_l)^T / B
          grad(b_l) = - mean(eps_{l+1})
        """
        eps = _compute_eps_disc(xs)
        B = xs[0].shape[0]
        invB = 1.0 / float(B)

        gradsV = []
        gradsb = []
        for l in range(L):
            pre = _act(xs[l])                 # [B, d_l]
            post_err = eps[l + 1]             # [B, d_{l+1}]
            # gradient of energy w.r.t V is - post_err^T @ pre / B
            gV = -(post_err.T @ pre) * invB   # [d_{l+1}, d_l]
            gb_ = -post_err.mean(dim=0)       # [d_{l+1}]
            gradsV.append(gV)
            gradsb.append(gb_)

        _param_step(gradsV, gradsb)

    def _predict_logits(x0, T):
        xs = _ff_init(x0)
        clamp = [True] + [False] * L  # clamp x1 only
        xs = _infer_disc(xs, clamp, T=T)
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

    def _mean_energy(loader, T, supervised=False):
        """
        Optional sanity: mean E_disc after inference.
        supervised=True clamps output to one-hot labels (like training equilibrium).
        """
        Es = []
        for xb, yb in loader:
            xs = _ff_init(xb)
            xs[0] = xb
            clamp = [True] + [False] * (L - 1) + [not supervised]  # if supervised: True
            if supervised:
                xs[L] = _one_hot(yb)
                clamp = [True] + [False] * (L - 1) + [True]
            xs = _infer_disc(xs, clamp, T=T)
            eps = _compute_eps_disc(xs)
            # Energy uses alpha_disc/2 ||x - pred||^2; since eps=alpha*(x-pred), energy = 1/(2*alpha) ||eps||^2
            # but alpha_disc could be 1; this is consistent up to constant scaling. We report raw sum of squared residuals:
            e = 0.0
            for l in range(1, L + 1):
                # residual = (eps/alpha_disc)
                res = eps[l] / max(alpha_disc, 1e-12)
                e += 0.5 * alpha_disc * (res * res).sum(dim=1).mean().item()
            Es.append(e)
        if not Es:
            return 0.0
        return float(sum(Es) / len(Es))

    # -------------------------
    # 3) data
    # -------------------------
    Xtr, ytr = _make_blobs(train_n)
    Xte, yte = _make_blobs(test_n)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)

    # -------------------------
    # 4) train loop (discPC)
    # -------------------------
    for ep in range(1, epochs + 1):
        for xb, yb in train_loader:
            xs = _ff_init(xb)

            # supervised clamp: x1 fixed to input, xL fixed to one-hot label (paper protocol)
            xs[0] = xb
            xs[L] = _one_hot(yb)
            clamp = [True] + [False] * (L - 1) + [True]

            xs = _infer_disc(xs, clamp, T=steps_train)
            _update_params_disc(xs)

        acc = _accuracy(test_loader, T=steps_test)
        results[f"epoch_{ep}/test_acc"] = float(acc)

        # optional monitoring
        results[f"epoch_{ep}/E_test_after_infer"] = _mean_energy(test_loader, T=steps_test, supervised=False)

    # sanity metrics
    with torch.no_grad():
        for l in range(L):
            results[f"V{l}_fro"] = float(torch.norm(V[l]).item())
            results[f"b{l}_l2"] = float(torch.norm(b[l]).item())

    return results
