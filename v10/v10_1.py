# v5_MNIST_bipc_trainable.py
# BiPCN / bPC (Bidirectional Predictive Coding) aligned to arXiv:2505.23415 Eq.(4)-(7)
#
# Key equations (paper):
#   E = sum_{l=1..L-1} αgen/2 || x_l - W_{l+1} f(x_{l+1}) ||^2
#     + sum_{l=2..L}   αdisc/2|| x_l - V_{l-1} f(x_{l-1}) ||^2        (Eq.4)
#   ε_gen_l  := αgen (x_l - W_{l+1} f(x_{l+1}))                      (Eq.6)
#   ε_disc_l := αdisc(x_l - V_{l-1} f(x_{l-1}))                      (Eq.6)
#   dx_l/dt = -ε_gen_l - ε_disc_l + f'(x_l) ⊙ ( W_l^T ε_gen_{l-1} + V_l^T ε_disc_{l+1} ) (Eq.5)
#   ΔW_l ∝ ε_gen_{l-1} f(x_l)^T ;  ΔV_l ∝ ε_disc_{l+1} f(x_l)^T      (Eq.7)
#
# 0-index mapping (layers 0..L):
#   - x0: input image, xL: label layer
#   - W[k]: top-down weight predicting x_k from x_{k+1} (shape dim_k x dim_{k+1})
#   - V[k]: bottom-up weight predicting x_{k+1} from x_k (shape dim_{k+1} x dim_k)
#   - ε_gen[k]  corresponds to layer k (k=0..L-1)
#   - ε_disc[k] corresponds to layer k (k=1..L)
# 在文件开头补充这两个 import（如果还没有）

import os
import radas  # pyright: ignore[reportMissingImports]

# --------------------------
# Data: MNIST scaled to [-1, 1]
# --------------------------
def load_mnist_data(
    batch_size=256,
    data_root=None,
    download=False,          # 关键：默认不下载
    num_workers=0,
    pin_memory=False,
    user_name="mengfan",     # 关键：用 radas 定位数据目录
):
    """
    从你已下载好的 MNIST 目录读取数据（不再下载）。
    你的存放路径应为：radas.get_data_dir(user_name)/data
    """
    if data_root is None:
        base_dir = radas.get_data_dir(user_name=user_name)
        data_root = os.path.join(base_dir, "data")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1,1]
    ])

    # download=False：如果目录下不存在 raw/processed，会直接报错提醒
    train_dataset = datasets.MNIST(root=data_root, train=True, download=download, transform=transform)
    test_dataset  = datasets.MNIST(root=data_root, train=False, download=download, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, test_loader

import math
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --------------------------
# Activations
# --------------------------
def _identity(x):
    return x

def _identity_prime(x):
    return torch.ones_like(x)

def _tanh(x):
    return torch.tanh(x)

def _tanh_prime(x):
    y = torch.tanh(x)
    return 1.0 - y * y

def _leaky_relu(x, negative_slope=0.01):
    return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)

def _leaky_relu_prime(x, negative_slope=0.01):
    return torch.where(x >= 0, torch.ones_like(x), torch.full_like(x, negative_slope))

def _gelu(x):
    return torch.nn.functional.gelu(x)

def _gelu_prime(x):
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    erf_term = torch.special.erf(x * inv_sqrt2)
    exp_term = torch.exp(-0.5 * x * x)
    return 0.5 * (1.0 + erf_term) + x * exp_term * inv_sqrt2pi

def get_activation(name: str, negative_slope: float = 0.01):
    name = (name or "leaky_relu").lower()
    if name in ["identity", "linear", "none"]:
        return _identity, _identity_prime
    if name == "tanh":
        return _tanh, _tanh_prime
    if name in ["leaky_relu", "lrelu"]:
        return (
            lambda x: _leaky_relu(x, negative_slope=negative_slope),
            lambda x: _leaky_relu_prime(x, negative_slope=negative_slope),
        )
    if name == "gelu":
        return _gelu, _gelu_prime
    raise ValueError(f"Unknown activation: {name}")



# --------------------------
# Minimal AdamW
# --------------------------
class AdamW:
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-3):
        self.params = params
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.state = []
        for p in params:
            self.state.append(dict(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p)))

    @torch.no_grad()
    def step(self, grads):
        for p, g, st in zip(self.params, grads, self.state):
            st["t"] += 1
            t = st["t"]
            st["m"].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            st["v"].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

            m_hat = st["m"] / (1.0 - (self.beta1 ** t))
            v_hat = st["v"] / (1.0 - (self.beta2 ** t))

            # decoupled weight decay
            if self.weight_decay > 0:
                p.mul_(1.0 - self.lr * self.weight_decay)

            p.addcdiv_(m_hat, torch.sqrt(v_hat).add_(self.eps), value=-self.lr)


# --------------------------
# BiPCN / bPC MLP (Eq.4-7)
# --------------------------
class BiPCN_MLP:
    def __init__(self, layers, activation="leaky_relu", use_bias=True, device="cpu",
                 negative_slope=0.01, weight_init="xavier"):
        self.layers = list(layers)
        self.L = len(self.layers) - 1
        self.use_bias = bool(use_bias)
        self.device = torch.device(device)

        self.f, self.fprime = get_activation(activation, negative_slope=negative_slope)

        # Bottom-up V[k]: dim_{k+1} x dim_k
        # Top-down  W[k]: dim_k x dim_{k+1}
        self.V = []
        self.bV = []
        self.W = []
        self.bW = []

        for k in range(self.L):
            d_k = self.layers[k]
            d_k1 = self.layers[k + 1]

            V = torch.empty(d_k1, d_k, device=self.device, dtype=torch.float32)
            W = torch.empty(d_k, d_k1, device=self.device, dtype=torch.float32)

            if weight_init == "xavier":
                # Xavier uniform
                boundV = math.sqrt(6.0 / (d_k + d_k1))
                boundW = math.sqrt(6.0 / (d_k + d_k1))
                V.uniform_(-boundV, boundV)
                W.uniform_(-boundW, boundW)
            else:
                V.normal_(mean=0.0, std=math.sqrt(1.0 / d_k))
                W.normal_(mean=0.0, std=math.sqrt(1.0 / d_k1))

            self.V.append(V)
            self.W.append(W)

            if self.use_bias:
                self.bV.append(torch.zeros(d_k1, 1, device=self.device, dtype=torch.float32))
                self.bW.append(torch.zeros(d_k, 1, device=self.device, dtype=torch.float32))
            else:
                self.bV.append(None)
                self.bW.append(None)

        self.opt_theta = None  # set by trainable

    def bottom_up_sweep(self, x0, clamp_output=False, xL=None):
        acts = [None] * (self.L + 1)
        acts[0] = x0
        for k in range(self.L):
            pred = self.V[k] @ self.f(acts[k])
            if self.use_bias and (self.bV[k] is not None):
                pred = pred + self.bV[k]
            acts[k + 1] = pred
        if clamp_output:
            if xL is None:
                raise ValueError("clamp_output=True but xL is None")
            acts[self.L] = xL
        return acts

    def top_down_sweep(self, xL, clamp_input=False, x0=None):
        acts = [None] * (self.L + 1)
        acts[self.L] = xL
        for k in reversed(range(self.L)):
            pred = self.W[k] @ self.f(acts[k + 1])  # predicts x_k from x_{k+1}
            if self.use_bias and (self.bW[k] is not None):
                pred = pred + self.bW[k]
            acts[k] = pred
        if clamp_input:
            if x0 is None:
                raise ValueError("clamp_input=True but x0 is None")
            acts[0] = x0
        return acts

    def compute_errors(self, acts, alpha_gen=1.0, alpha_disc=1.0):
        # eps_gen[k]  for k=0..L-1
        # eps_disc[k] for k=1..L
        eps_gen = [None] * (self.L + 1)
        eps_disc = [None] * (self.L + 1)

        # generative: x_k - W[k] f(x_{k+1})
        for k in range(self.L):
            pred = self.W[k] @ self.f(acts[k + 1])
            if self.use_bias and (self.bW[k] is not None):
                pred = pred + self.bW[k]
            eps_gen[k] = float(alpha_gen) * (acts[k] - pred)

        # discriminative: x_k - V[k-1] f(x_{k-1})
        for k in range(1, self.L + 1):
            pred = self.V[k - 1] @ self.f(acts[k - 1])
            if self.use_bias and (self.bV[k - 1] is not None):
                pred = pred + self.bV[k - 1]
            eps_disc[k] = float(alpha_disc) * (acts[k] - pred)

        return eps_gen, eps_disc

    @torch.no_grad()
    def energy(self, acts, alpha_gen=1.0, alpha_disc=1.0):
        # Return (E_total, E_gen, E_disc) in "paper energy" scale:
        # E_gen = sum αgen/2 ||x_k - W[k]f(x_{k+1})||^2 ; E_disc = sum αdisc/2 ||x_k - V[k-1]f(x_{k-1})||^2
        B = acts[0].shape[1]
        E_gen = 0.0
        E_disc = 0.0

        # gen terms for k=0..L-1
        for k in range(self.L):
            pred = self.W[k] @ self.f(acts[k + 1])
            if self.use_bias and (self.bW[k] is not None):
                pred = pred + self.bW[k]
            diff = acts[k] - pred
            E_gen += 0.5 * float(alpha_gen) * float((diff * diff).sum().item()) / max(1, B)

        # disc terms for k=1..L
        for k in range(1, self.L + 1):
            pred = self.V[k - 1] @ self.f(acts[k - 1])
            if self.use_bias and (self.bV[k - 1] is not None):
                pred = pred + self.bV[k - 1]
            diff = acts[k] - pred
            E_disc += 0.5 * float(alpha_disc) * float((diff * diff).sum().item()) / max(1, B)

        return (E_gen + E_disc), E_gen, E_disc

    @torch.no_grad()
    def infer(
        self,
        acts,
        steps,
        lr_x=0.01,
        momentum_x=0.0,
        alpha_gen=1.0,
        alpha_disc=1.0,
        clamp_input=True,
        x0=None,
        clamp_output=False,
        xL=None,
        # optional partial clamp on input (for missing pixels)
        input_clamp_mask=None,  # shape (dim0,1) or (dim0,B), True=clamped
    ):
        # momentum buffers for x1..xL (and optionally x0)
        vel = [None] * (self.L + 1)
        for k in range(self.L + 1):
            if k == 0:
                vel[k] = torch.zeros_like(acts[k])
            else:
                vel[k] = torch.zeros_like(acts[k])

        for _ in range(int(steps)):
            eps_gen, eps_disc = self.compute_errors(acts, alpha_gen=alpha_gen, alpha_disc=alpha_disc)

            # --- update x0 (input) if not fully clamped ---
            if not clamp_input:
                # Eq.(5) boundary for x0: dx0 = -ε_gen0 + f'(x0) ⊙ ( V0^T ε_disc1 )
                dx0 = -eps_gen[0]
                dx0 = dx0 + self.fprime(acts[0]) * (self.V[0].T @ eps_disc[1])
                vel[0].mul_(momentum_x).add_(dx0)
                acts[0].add_(vel[0], alpha=lr_x)

            # partial clamp for x0 (missing pixel protocol)
            if input_clamp_mask is not None:
                if x0 is None:
                    raise ValueError("input_clamp_mask is set but x0 (observed values) is None")
                # clamp where mask=True, free where mask=False
                acts[0] = torch.where(input_clamp_mask, x0, acts[0])

            # full clamp for x0
            if clamp_input:
                if x0 is None:
                    raise ValueError("clamp_input=True but x0 is None")
                acts[0] = x0

            # --- update hidden layers x1..x_{L-1} ---
            for k in range(1, self.L):
                # Eq.(5): dxk = -ε_genk - ε_disck + f'(xk) ⊙ ( W_{k}^T ε_gen_{k-1} + V_{k}^T ε_disc_{k+1} )
                # (0-index: W[k-1] predicts x_{k-1} from x_k, so W[k-1]^T maps ε_gen[k-1] -> layer k)
                msg = (self.W[k - 1].T @ eps_gen[k - 1]) + (self.V[k].T @ eps_disc[k + 1])
                dxk = -(eps_gen[k] + eps_disc[k]) + self.fprime(acts[k]) * msg
                vel[k].mul_(momentum_x).add_(dxk)
                acts[k].add_(vel[k], alpha=lr_x)

            # --- update output xL if not clamped ---
            if not clamp_output:
                # boundary Eq.(5) for xL: dxL = -ε_discL + f'(xL) ⊙ ( W_{L-1}^T ε_gen_{L-1} )
                msgL = self.W[self.L - 1].T @ eps_gen[self.L - 1]
                dxL = -eps_disc[self.L] + self.fprime(acts[self.L]) * msgL
                vel[self.L].mul_(momentum_x).add_(dxL)
                acts[self.L].add_(vel[self.L], alpha=lr_x)

            if clamp_output:
                if xL is None:
                    raise ValueError("clamp_output=True but xL is None")
                acts[self.L] = xL

        return acts

    @torch.no_grad()
    def update_theta_local(self, acts, alpha_gen=1.0, alpha_disc=1.0):
        # Eq.(7): ΔW_l ∝ ε_gen_{l-1} f(x_l)^T  ; ΔV_l ∝ ε_disc_{l+1} f(x_l)^T
        eps_gen, eps_disc = self.compute_errors(acts, alpha_gen=alpha_gen, alpha_disc=alpha_disc)
        B = acts[0].shape[1]

        grads = []
        params = []

        # parameters order: [W0,bW0,W1,bW1,..., V0,bV0,V1,bV1,...]
        for k in range(self.L):
            # W[k] predicts x_k from x_{k+1}
            # gradient of E wrt W[k] is - eps_gen[k] f(x_{k+1})^T / B (eps already includes αgen)
            fx_up = self.f(acts[k + 1])
            gW = -(eps_gen[k] @ fx_up.T) / max(1, B)
            grads.append(gW)
            params.append(self.W[k])

            if self.use_bias and (self.bW[k] is not None):
                gb = -eps_gen[k].mean(dim=1, keepdim=True)
                grads.append(gb)
                params.append(self.bW[k])

        for k in range(self.L):
            # V[k] predicts x_{k+1} from x_k
            fx_dn = self.f(acts[k])
            gV = -(eps_disc[k + 1] @ fx_dn.T) / max(1, B)
            grads.append(gV)
            params.append(self.V[k])

            if self.use_bias and (self.bV[k] is not None):
                gb = -eps_disc[k + 1].mean(dim=1, keepdim=True)
                grads.append(gb)
                params.append(self.bV[k])

        if self.opt_theta is None:
            raise RuntimeError("opt_theta is not set")
        self.opt_theta.step(grads)

    @torch.no_grad()
    def predict_class(self, x0, steps_eval=100, lr_x=0.01, momentum_x=0.0, alpha_gen=1.0, alpha_disc=1.0):
        # Paper classification: clamp x1(image), bottom-up init, 100 inference steps, argmax(xL) :contentReference[oaicite:6]{index=6}
        acts = self.bottom_up_sweep(x0, clamp_output=False, xL=None)
        acts = self.infer(
            acts, steps=steps_eval, lr_x=lr_x, momentum_x=momentum_x,
            alpha_gen=alpha_gen, alpha_disc=alpha_disc,
            clamp_input=True, x0=x0,
            clamp_output=False, xL=None
        )
        return torch.argmax(acts[self.L], dim=0)

    @torch.no_grad()
    def generate_from_label(self, y_onehot, steps_eval=100, lr_x=0.01, momentum_x=0.0,
                            alpha_gen=1.0, alpha_disc=1.0, init="topdown"):
        # For generation, paper clamps xL to label; for models with top-down pathway we can top-down init :contentReference[oaicite:7]{index=7}
        if init == "topdown":
            acts = self.top_down_sweep(y_onehot, clamp_input=False)
        else:
            # fallback: set input to zeros and bottom-up sweep (not ideal for bPC, but kept for debugging)
            B = y_onehot.shape[1]
            x0 = torch.zeros(self.layers[0], B, device=self.device)
            acts = self.bottom_up_sweep(x0, clamp_output=True, xL=y_onehot)

        acts = self.infer(
            acts, steps=steps_eval, lr_x=lr_x, momentum_x=momentum_x,
            alpha_gen=alpha_gen, alpha_disc=alpha_disc,
            clamp_input=False, x0=None,
            clamp_output=True, xL=y_onehot
        )
        return acts[0]  # generated x0 (image space)


# --------------------------
# trainable(config)
# --------------------------
def trainable(config: dict):
    results = {}

    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_cfg = config.get("device", None)
    device = torch.device(device_cfg) if device_cfg is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results["device"] = str(device)
    results["torch.cuda.is_available"] = bool(torch.cuda.is_available())

    # data
    # data
    batch_size = int(config.get("batch_size", 256))

    # 你也可以继续允许外部传 data_root；不传就用 radas 默认目录
    data_root = config.get("data_root", None)

    download = bool(config.get("download", False))  # 关键：默认 False
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

    # 可选：打印一下最终使用的数据目录，方便你确认
    print(f"[MNIST] data_root={data_root if data_root is not None else os.path.join(radas.get_data_dir(user_name=user_name), 'data')}, download={download}")


    # model + hyperparams (paper-like defaults)
    layers = config.get("layers", [784, 256, 256, 10])
    activation = config.get("activation", "leaky_relu")
    negative_slope = float(config.get("negative_slope", 0.01))
    use_bias = bool(config.get("use_bias", True))
    weight_init = config.get("weight_init", "xavier")

    # inference
    steps_train = int(config.get("steps_train", 8))     # T :contentReference[oaicite:8]{index=8}
    steps_test  = int(config.get("steps_test", 100))    # T_eval :contentReference[oaicite:9]{index=9}
    lr_x = float(config.get("lr_x", 0.01))
    momentum_x = float(config.get("momentum_x", 0.0))

    # energy weights αgen, αdisc (Eq.4) :contentReference[oaicite:10]{index=10}
    alpha_gen  = float(config.get("alpha_gen", 1.0))
    alpha_disc = float(config.get("alpha_disc", 1.0))

    # theta optimiser (AdamW on both W and V)
    lr_theta = float(config.get("lr_theta", 1e-4))
    weight_decay = float(config.get("weight_decay", 5e-3))
    adam_beta1 = float(config.get("adam_beta1", 0.9))
    adam_beta2 = float(config.get("adam_beta2", 0.999))
    adam_eps = float(config.get("adam_eps", 1e-8))

    n_epochs = int(config.get("epochs", 25))
    log_every = int(config.get("log_every", 5))

    # optional: compute class-average generation RMSE (Table 10 style) 
    eval_gen_rmse = bool(config.get("eval_gen_rmse", False))
    gen_steps = int(config.get("gen_steps", steps_test))
    gen_init = str(config.get("gen_init", "topdown"))

    # label encoding
    label_mode = str(config.get("label_mode", "onehot_01")).lower()  # onehot_01 or onehot_pm1

    # record config
    results["config/seed"] = seed
    results["config/batch_size"] = batch_size
    results["config/epochs"] = n_epochs
    results["config/layers"] = str(layers)
    results["config/activation"] = str(activation)
    results["config/steps_train"] = steps_train
    results["config/steps_test"] = steps_test
    results["config/lr_x"] = lr_x
    results["config/momentum_x"] = momentum_x
    results["config/lr_theta"] = lr_theta
    results["config/weight_decay"] = weight_decay
    results["config/alpha_gen"] = alpha_gen
    results["config/alpha_disc"] = alpha_disc
    results["config/label_mode"] = label_mode
    results["config/eval_gen_rmse"] = eval_gen_rmse

    model = BiPCN_MLP(
        layers=layers,
        activation=activation,
        use_bias=use_bias,
        device=device,
        negative_slope=negative_slope,
        weight_init=weight_init,
    )

    # AdamW parameter list: [W0,bW0,...,V0,bV0,...] in same order as update_theta_local builds grads
    theta_params = []
    for k in range(model.L):
        theta_params.append(model.W[k])
        if model.use_bias and (model.bW[k] is not None):
            theta_params.append(model.bW[k])
    for k in range(model.L):
        theta_params.append(model.V[k])
        if model.use_bias and (model.bV[k] is not None):
            theta_params.append(model.bV[k])

    model.opt_theta = AdamW(theta_params, lr=lr_theta, betas=(adam_beta1, adam_beta2), eps=adam_eps, weight_decay=weight_decay)

    def make_label_onehot(targets: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.one_hot(targets, num_classes=10).to(torch.float32)  # (B,10)
        if label_mode == "onehot_pm1":
            y = 2.0 * y - 1.0
        return y.T  # (10,B)

    # precompute class-average images for RMSE (optional)
    class_avg = None
    if eval_gen_rmse:
        # build from test set (paper: class-average image from eval set) :contentReference[oaicite:12]{index=12}
        sums = torch.zeros(10, 784, device=device)
        cnts = torch.zeros(10, device=device)
        with torch.no_grad():
            for data, targets in test_loader:
                x = data.view(data.size(0), -1).to(device=device, dtype=torch.float32)  # (B,784)
                for c in range(10):
                    mask = (targets == c)
                    if mask.any():
                        sums[c] += x[mask].sum(dim=0)
                        cnts[c] += mask.sum()
        class_avg = (sums / cnts.unsqueeze(1).clamp_min(1.0)).view(10, 784)  # still in [-1,1] space

    for epoch in range(n_epochs):
        E_sum = 0.0
        Eg_sum = 0.0
        Ed_sum = 0.0
        n_batches = 0

        for data, targets in train_loader:
            x0 = data.view(data.size(0), -1).to(device=device, dtype=torch.float32).T  # (784,B)
            y  = make_label_onehot(targets.to(device=device))

            # init as paper: bottom-up sweep (even for bPC) :contentReference[oaicite:13]{index=13}
            acts = model.bottom_up_sweep(x0, clamp_output=True, xL=y)

            # inference with clamped ends
            acts = model.infer(
                acts, steps=steps_train, lr_x=lr_x, momentum_x=momentum_x,
                alpha_gen=alpha_gen, alpha_disc=alpha_disc,
                clamp_input=True, x0=x0,
                clamp_output=True, xL=y
            )

            # local parameter update Eq.(7)
            model.update_theta_local(acts, alpha_gen=alpha_gen, alpha_disc=alpha_disc)

            Et, Eg, Ed = model.energy(acts, alpha_gen=alpha_gen, alpha_disc=alpha_disc)
            E_sum += Et
            Eg_sum += Eg
            Ed_sum += Ed
            n_batches += 1

        E_train = E_sum / max(1, n_batches)
        Egen_train = Eg_sum / max(1, n_batches)
        Edisc_train = Ed_sum / max(1, n_batches)

        # --- eval: test_acc + clamped energy ---
        test_correct = 0
        test_total = 0
        Etest_sum = 0.0
        Egtest_sum = 0.0
        Edtest_sum = 0.0
        Etest_batches = 0

        with torch.no_grad():
            for data, targets in test_loader:
                x0 = data.view(data.size(0), -1).to(device=device, dtype=torch.float32).T
                tdev = targets.to(device=device)
                y = make_label_onehot(tdev)

                pred = model.predict_class(
                    x0, steps_eval=steps_test, lr_x=lr_x, momentum_x=momentum_x,
                    alpha_gen=alpha_gen, alpha_disc=alpha_disc
                )
                test_correct += int((pred == tdev).sum().item())
                test_total += int(tdev.numel())

                # clamped energy after inference (monitor)
                acts = model.bottom_up_sweep(x0, clamp_output=True, xL=y)
                acts = model.infer(
                    acts, steps=steps_test, lr_x=lr_x, momentum_x=momentum_x,
                    alpha_gen=alpha_gen, alpha_disc=alpha_disc,
                    clamp_input=True, x0=x0,
                    clamp_output=True, xL=y
                )
                Et, Eg, Ed = model.energy(acts, alpha_gen=alpha_gen, alpha_disc=alpha_disc)
                Etest_sum += Et
                Egtest_sum += Eg
                Edtest_sum += Ed
                Etest_batches += 1

        test_acc = test_correct / max(1, test_total)
        E_test = Etest_sum / max(1, Etest_batches)
        Egen_test = Egtest_sum / max(1, Etest_batches)
        Edisc_test = Edtest_sum / max(1, Etest_batches)

        # optional: class-average generation RMSE (Table 10 style)
        gen_rmse = None
        if eval_gen_rmse:
            # generate one image per class (batch=10)
            ys = torch.eye(10, device=device, dtype=torch.float32).T  # (10,10) each col is one-hot
            if label_mode == "onehot_pm1":
                ys = 2.0 * ys - 1.0

            x_gen = model.generate_from_label(
                ys, steps_eval=gen_steps, lr_x=lr_x, momentum_x=momentum_x,
                alpha_gen=alpha_gen, alpha_disc=alpha_disc, init=gen_init
            )  # (784,10)
            x_gen = x_gen.T  # (10,784)
            rmse = torch.sqrt(torch.mean((x_gen - class_avg) ** 2)).item()
            gen_rmse = float(rmse)

        ep = epoch + 1
        results[f"epoch_{ep}/E_train_after_infer"] = float(E_train)
        results[f"epoch_{ep}/Egen_train_after_infer"] = float(Egen_train)
        results[f"epoch_{ep}/Edisc_train_after_infer"] = float(Edisc_train)

        results[f"epoch_{ep}/test_acc"] = float(test_acc)
        results[f"epoch_{ep}/E_test_after_infer"] = float(E_test)
        results[f"epoch_{ep}/Egen_test_after_infer"] = float(Egen_test)
        results[f"epoch_{ep}/Edisc_test_after_infer"] = float(Edisc_test)

        if eval_gen_rmse:
            results[f"epoch_{ep}/gen_rmse_classavg"] = gen_rmse

        if log_every > 0 and (ep % log_every == 0 or ep == 1 or ep == n_epochs):
            msg = (f"Epoch {ep:3d} | test_acc={test_acc:.4f} | "
                   f"E_test={E_test:.4f} (gen={Egen_test:.4f}, disc={Edisc_test:.4f})")
            if eval_gen_rmse:
                msg += f" | gen_rmse={gen_rmse:.4f}"
            print(msg)

    results["final/test_acc"] = results.get(f"epoch_{n_epochs}/test_acc", None)
    results["final/E_train_after_infer"] = results.get(f"epoch_{n_epochs}/E_train_after_infer", None)
    results["final/E_test_after_infer"] = results.get(f"epoch_{n_epochs}/E_test_after_infer", None)
    if eval_gen_rmse:
        results["final/gen_rmse_classavg"] = results.get(f"epoch_{n_epochs}/gen_rmse_classavg", None)

    return results


if __name__ == "__main__":
    cfg = dict(
        seed=0,
        device=None,
        data_root="./data",
        download=True,
        batch_size=256,
        epochs=100,
        layers=[784, 256, 256, 10],
        activation="leaky_relu",
        steps_train=8,
        steps_test=100,
        lr_x=0.01,
        momentum_x=0.0,
        lr_theta=1e-4,
        weight_decay=5e-3,
        alpha_gen=0.01,
        alpha_disc=1.0,
        label_mode="onehot_01",
        eval_gen_rmse=False,
        log_every=5,
    )
    out = trainable(cfg)
    print("final/test_acc =", out["final/test_acc"])
