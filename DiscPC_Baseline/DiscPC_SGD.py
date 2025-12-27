# v5_MNIST_discpc_trainable.py
# DiscPC (as in arXiv:2505.23415v1) for MNIST classification

import os
import math
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import radas  # pyright: ignore[reportMissingImports]

# --------------------------
# Activations for "sending" f(x_l) used in predictions V_l f(x_l)
# --------------------------
def _identity(x):
    return x


def _identity_prime(x):  # noqa: ARG001
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
    # gelu(x) = 0.5 x (1 + erf(x/sqrt(2)))
    # d/dx = 0.5 (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2π)
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
# Data loader: MNIST scaled to [-1, 1]
# --------------------------
def load_mnist_data(
    batch_size=256,
    data_root=None,
    download=False,
    num_workers=0,
    pin_memory=False,
    user_name="mengfan",
):
    if data_root is None:
        base_dir = radas.get_data_dir(user_name=user_name)
        data_root = os.path.join(base_dir, "data")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1,1]
        ]
    )

    train_dataset = datasets.MNIST(root=data_root, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=download, transform=transform)

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
    return train_loader, test_loader


# --------------------------
# LR helpers
# --------------------------
class ConstantLR:
    def __init__(self, lr: float):
        self.lr = float(lr)

    def lr_at(self, step: int) -> float:  # noqa: ARG002
        return self.lr


class CosineLR:
    def __init__(self, base_lr: float, total_steps: int, min_lr: float = 0.0):
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.total_steps = int(max(1, total_steps))

    def lr_at(self, step: int) -> float:
        s = int(max(0, min(step, self.total_steps)))
        t = float(s) / float(self.total_steps)
        t = max(0.0, min(1.0, t))
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


class CosineWarmupLR:
    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int = 0, min_lr: float = 0.0):
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.total_steps = int(max(1, total_steps))
        self.warmup_steps = int(max(0, warmup_steps))
        if self.warmup_steps > self.total_steps:
            self.warmup_steps = self.total_steps

    def lr_at(self, step: int) -> float:
        s = int(max(0, min(step, self.total_steps)))
        if self.warmup_steps > 0 and s < self.warmup_steps:
            # linear warmup: 0 -> base_lr
            return self.base_lr * (float(s + 1) / float(self.warmup_steps))

        if self.total_steps == self.warmup_steps:
            return self.min_lr

        t = float(s - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
        t = max(0.0, min(1.0, t))
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


# --------------------------
# Minimal SGD (manual) for local gradients
# - mini-batch: grads computed from current batch, averaged by B
# - supports momentum
# - decoupled weight decay (optional), can skip bias
# --------------------------
class SGD:
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0, apply_wd=None):
        self.params = list(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)

        if apply_wd is None:
            self.apply_wd = [True for _ in self.params]
        else:
            if len(apply_wd) != len(self.params):
                raise ValueError("apply_wd must have same length as params")
            self.apply_wd = [bool(x) for x in apply_wd]

        self.state = []
        for p in self.params:
            self.state.append(dict(v=torch.zeros_like(p)))

    def set_lr(self, lr: float):
        self.lr = float(lr)

    @torch.no_grad()
    def step(self, grads):
        for p, g, st, do_wd in zip(self.params, grads, self.state, self.apply_wd):
            if do_wd and self.weight_decay > 0.0:
                p.mul_(1.0 - self.lr * self.weight_decay)

            if self.momentum > 0.0:
                st["v"].mul_(self.momentum).add_(g)
                p.add_(st["v"], alpha=-self.lr)
            else:
                p.add_(g, alpha=-self.lr)


# --------------------------
# DiscPC MLP (bottom-up only)
# --------------------------
class DiscPCMLP:
    def __init__(
        self,
        layers,
        activation="leaky_relu",
        last_send_activation="identity",
        use_bias=True,
        device="cpu",
        negative_slope=0.01,
        weight_init="xavier",
    ):
        self.layers = list(layers)
        self.L = len(self.layers) - 1
        self.use_bias = bool(use_bias)
        self.device = torch.device(device)

        f, fprime = get_activation(activation, negative_slope=negative_slope)
        f_last, fprime_last = get_activation(last_send_activation, negative_slope=negative_slope)

        self.send_f = [f for _ in range(self.L)]
        self.send_fprime = [fprime for _ in range(self.L)]
        self.send_f[-1] = f_last
        self.send_fprime[-1] = fprime_last

        self.weights = []
        self.biases = []
        for l in range(self.L):
            din = self.layers[l]
            dout = self.layers[l + 1]
            W = torch.empty(din, dout, device=self.device, dtype=torch.float32)

            if weight_init == "xavier":
                bound = math.sqrt(6.0 / (din + dout))
                W.uniform_(-bound, bound)
            else:
                W.normal_(mean=0.0, std=math.sqrt(1.0 / din))

            self.weights.append(W)

            if self.use_bias:
                b = torch.zeros(dout, 1, device=self.device, dtype=torch.float32)
                self.biases.append(b)
            else:
                self.biases.append(None)

        self.opt_theta = None

    def _predict_next(self, l, x_l):
        z = self.weights[l].T @ self.send_f[l](x_l)
        if self.use_bias and (self.biases[l] is not None):
            z = z + self.biases[l]
        return z

    def init_bottom_up(self, x0, clamp_output=False, xL=None):
        acts = [None] * (self.L + 1)
        acts[0] = x0

        for l in range(self.L - 1):
            acts[l + 1] = self._predict_next(l, acts[l])

        if clamp_output:
            if xL is None:
                raise ValueError("clamp_output=True but xL is None")
            acts[self.L] = xL
        else:
            acts[self.L] = self._predict_next(self.L - 1, acts[self.L - 1])

        return acts

    def preds_and_eps(self, acts):
        preds = [None] * (self.L + 1)
        eps = [None] * (self.L + 1)
        for k in range(1, self.L + 1):
            preds[k] = self._predict_next(k - 1, acts[k - 1])
            eps[k] = acts[k] - preds[k]
        return preds, eps

    @torch.no_grad()
    def infer(
        self,
        acts,
        steps,
        lr_x=0.01,
        momentum_x=0.0,
        clamp_input=True,
        x0=None,
        clamp_output=False,
        xL=None,
    ):
        vel = [None] * (self.L + 1)
        for k in range(1, self.L + 1):
            vel[k] = torch.zeros_like(acts[k])

        for _ in range(int(steps)):
            _, eps = self.preds_and_eps(acts)

            for l in range(1, self.L):
                dx = -eps[l] + self.send_fprime[l](acts[l]) * (self.weights[l] @ eps[l + 1])
                vel[l].mul_(momentum_x).add_(dx)
                acts[l].add_(vel[l], alpha=lr_x)

            if not clamp_output:
                dxL = -eps[self.L]
                vel[self.L].mul_(momentum_x).add_(dxL)
                acts[self.L].add_(vel[self.L], alpha=lr_x)

            if clamp_input:
                if x0 is None:
                    raise ValueError("clamp_input=True but x0 is None")
                acts[0] = x0
            if clamp_output:
                if xL is None:
                    raise ValueError("clamp_output=True but xL is None")
                acts[self.L] = xL

        return acts

    @torch.no_grad()
    def energy(self, acts):
        _, eps = self.preds_and_eps(acts)
        B = acts[0].shape[1]
        e = 0.0
        for k in range(1, self.L + 1):
            e += 0.5 * float((eps[k] * eps[k]).sum().item()) / max(1, B)
        return e

    @staticmethod
    def _clip_grads_(grads, max_norm: float):
        if max_norm is None or max_norm <= 0:
            return
        total = 0.0
        for g in grads:
            total += float((g * g).sum().item())
        total_norm = math.sqrt(max(0.0, total))
        if total_norm <= max_norm:
            return
        scale = max_norm / (total_norm + 1e-12)
        for i in range(len(grads)):
            grads[i].mul_(scale)

    @torch.no_grad()
    def update_theta_local(self, acts, grad_clip_norm: float = 0.0):
        _, eps = self.preds_and_eps(acts)
        B = acts[0].shape[1]

        grads = []
        for l in range(self.L):
            fx = self.send_f[l](acts[l])
            gradW = -(fx @ eps[l + 1].T) / max(1, B)
            grads.append(gradW)

            if self.use_bias and (self.biases[l] is not None):
                gradb = -eps[l + 1].mean(dim=1, keepdim=True)
                grads.append(gradb)

        if grad_clip_norm and grad_clip_norm > 0.0:
            self._clip_grads_(grads, float(grad_clip_norm))

        if self.opt_theta is None:
            raise RuntimeError("opt_theta is not set")
        self.opt_theta.step(grads)

    @torch.no_grad()
    def predict_class(self, x0, steps_eval=100, lr_x=0.01, momentum_x=0.0):
        acts = self.init_bottom_up(x0, clamp_output=False, xL=None)
        acts = self.infer(
            acts,
            steps=steps_eval,
            lr_x=lr_x,
            momentum_x=momentum_x,
            clamp_input=True,
            x0=x0,
            clamp_output=False,
            xL=None,
        )
        logits = acts[self.L]
        return torch.argmax(logits, dim=0)


# --------------------------
# steps_train schedule (optional)
# --------------------------
def steps_train_for_epoch(epoch_idx: int, n_epochs: int, steps_init: int, steps_max: int, mode: str = "constant"):
    mode = (mode or "constant").lower()
    steps_init = int(steps_init)
    steps_max = int(steps_max)
    if mode == "constant" or steps_max <= steps_init or n_epochs <= 1:
        return steps_init
    if mode != "linear":
        raise ValueError(f"Unknown steps_train_schedule: {mode}")
    t = float(epoch_idx) / float(max(1, n_epochs - 1))
    return int(round(steps_init + (steps_max - steps_init) * t))


# --------------------------
# trainable(config)
# --------------------------
def trainable(config: dict):
    results = {}

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    batch_size = int(config.get("batch_size", 256))
    user_name = str(config.get("user_name", "mengfan"))
    data_root = config.get("data_root", None)
    download = bool(config.get("download", False))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", False))

    train_loader, test_loader = load_mnist_data(
        batch_size=batch_size,
        data_root=data_root,
        download=download,
        num_workers=num_workers,
        pin_memory=pin_memory,
        user_name=user_name,
    )
    print(f"[MNIST] root={data_root if data_root is not None else os.path.join(radas.get_data_dir(user_name=user_name), 'data')} download={download}")

    # model
    layers = config.get("layers", [784, 256, 256, 10])
    activation = config.get("activation", "leaky_relu")
    last_send_activation = config.get("last_send_activation", "identity")
    negative_slope = float(config.get("negative_slope", 0.01))
    use_bias = bool(config.get("use_bias", True))
    weight_init = config.get("weight_init", "xavier")

    # inference
    steps_train = int(config.get("steps_train", 8))
    steps_train_max = int(config.get("steps_train_max", steps_train))
    steps_train_schedule = str(config.get("steps_train_schedule", "constant")).lower()
    steps_test = int(config.get("steps_test", 100))
    lr_x = float(config.get("lr_x", 0.01))
    momentum_x = float(config.get("momentum_x", 0.0))

    # theta SGD
    lr_theta = float(config.get("lr_theta", 1e-3))
    momentum_theta = float(config.get("momentum_theta", 0.0))
    weight_decay = float(config.get("weight_decay", 0.0))
    grad_clip_norm = float(config.get("grad_clip_norm", 0.0))

    # lr schedule for theta (IMPORTANT: default constant, so your搜索不会“暗中变了训练”)
    lr_theta_schedule = str(config.get("lr_theta_schedule", "constant")).lower()  # constant|cosine|cosine_warmup
    lr_theta_min = float(config.get("lr_theta_min", 0.0))
    lr_theta_warmup_steps = int(config.get("lr_theta_warmup_steps", 0))

    n_epochs = int(config.get("epochs", 25))
    log_every = int(config.get("log_every", 5))

    label_mode = str(config.get("label_mode", "onehot_01")).lower()

    # record config
    results["config/seed"] = seed
    results["config/batch_size"] = batch_size
    results["config/epochs"] = n_epochs
    results["config/layers"] = str(layers)
    results["config/activation"] = str(activation)
    results["config/last_send_activation"] = str(last_send_activation)
    results["config/steps_train"] = steps_train
    results["config/steps_train_max"] = steps_train_max
    results["config/steps_train_schedule"] = steps_train_schedule
    results["config/steps_test"] = steps_test
    results["config/lr_x"] = lr_x
    results["config/momentum_x"] = momentum_x
    results["config/lr_theta"] = lr_theta
    results["config/momentum_theta"] = momentum_theta
    results["config/weight_decay"] = weight_decay
    results["config/grad_clip_norm"] = grad_clip_norm
    results["config/lr_theta_schedule"] = lr_theta_schedule
    results["config/lr_theta_min"] = lr_theta_min
    results["config/lr_theta_warmup_steps"] = lr_theta_warmup_steps
    results["config/label_mode"] = label_mode

    model = DiscPCMLP(
        layers=layers,
        activation=activation,
        last_send_activation=last_send_activation,
        use_bias=use_bias,
        device=device,
        negative_slope=negative_slope,
        weight_init=weight_init,
    )

    # attach SGD; skip weight decay on bias
    theta_params = []
    apply_wd = []
    for l in range(model.L):
        theta_params.append(model.weights[l])
        apply_wd.append(True)
        if model.use_bias and (model.biases[l] is not None):
            theta_params.append(model.biases[l])
            apply_wd.append(False)

    model.opt_theta = SGD(
        theta_params,
        lr=lr_theta,
        momentum=momentum_theta,
        weight_decay=weight_decay,
        apply_wd=apply_wd,
    )

    total_steps = n_epochs * max(1, len(train_loader))
    if lr_theta_schedule == "constant":
        lr_sched = ConstantLR(lr_theta)
    elif lr_theta_schedule == "cosine":
        lr_sched = CosineLR(base_lr=lr_theta, total_steps=total_steps, min_lr=lr_theta_min)
    elif lr_theta_schedule == "cosine_warmup":
        lr_sched = CosineWarmupLR(
            base_lr=lr_theta,
            total_steps=total_steps,
            warmup_steps=lr_theta_warmup_steps,
            min_lr=lr_theta_min,
        )
    else:
        raise ValueError(f"Unknown lr_theta_schedule: {lr_theta_schedule}")

    def make_label_onehot(targets: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.one_hot(targets, num_classes=10).to(torch.float32)
        if label_mode == "onehot_pm1":
            y = 2.0 * y - 1.0
        return y.T

    global_step = 0

    for epoch in range(n_epochs):
        model_energy_sum = 0.0
        n_batches = 0

        steps_train_ep = steps_train_for_epoch(
            epoch_idx=epoch,
            n_epochs=n_epochs,
            steps_init=steps_train,
            steps_max=steps_train_max,
            mode=steps_train_schedule,
        )

        for data, targets in train_loader:
            cur_lr = lr_sched.lr_at(global_step)
            model.opt_theta.set_lr(cur_lr)
            global_step += 1

            x0 = data.view(data.size(0), -1).to(device=device, dtype=torch.float32).T
            y = make_label_onehot(targets.to(device=device))

            acts = model.init_bottom_up(x0, clamp_output=True, xL=y)
            acts = model.infer(
                acts,
                steps=steps_train_ep,
                lr_x=lr_x,
                momentum_x=momentum_x,
                clamp_input=True,
                x0=x0,
                clamp_output=True,
                xL=y,
            )

            model.update_theta_local(acts, grad_clip_norm=grad_clip_norm)

            model_energy_sum += model.energy(acts)
            n_batches += 1

        E_train = model_energy_sum / max(1, n_batches)

        test_correct = 0
        test_total = 0
        E_test_sum = 0.0
        E_test_batches = 0

        with torch.no_grad():
            for data, targets in test_loader:
                x0 = data.view(data.size(0), -1).to(device=device, dtype=torch.float32).T
                targets_dev = targets.to(device=device)
                y = make_label_onehot(targets_dev)

                pred = model.predict_class(
                    x0,
                    steps_eval=steps_test,
                    lr_x=lr_x,
                    momentum_x=momentum_x,
                )
                test_correct += int((pred == targets_dev).sum().item())
                test_total += int(targets_dev.numel())

                acts = model.init_bottom_up(x0, clamp_output=True, xL=y)
                acts = model.infer(
                    acts,
                    steps=steps_test,
                    lr_x=lr_x,
                    momentum_x=momentum_x,
                    clamp_input=True,
                    x0=x0,
                    clamp_output=True,
                    xL=y,
                )
                E_test_sum += model.energy(acts)
                E_test_batches += 1

        test_acc = test_correct / max(1, test_total)
        E_test = E_test_sum / max(1, E_test_batches)

        ep = epoch + 1
        results[f"epoch_{ep}/steps_train_ep"] = int(steps_train_ep)
        results[f"epoch_{ep}/lr_theta_last"] = float(model.opt_theta.lr)
        results[f"epoch_{ep}/E_train_after_infer"] = float(E_train)
        results[f"epoch_{ep}/test_acc"] = float(test_acc)
        results[f"epoch_{ep}/E_test_after_infer"] = float(E_test)

        if log_every > 0 and (ep % log_every == 0 or ep == 1 or ep == n_epochs):
            print(
                f"Epoch {ep:3d} | steps_train={steps_train_ep:3d} | lr_theta(last)={model.opt_theta.lr:.6g} | "
                f"E_train={E_train:.6f} | test_acc={test_acc:.4f} | E_test(clamped)={E_test:.6f}"
            )

    results["final/test_acc"] = results.get(f"epoch_{n_epochs}/test_acc", None)
    results["final/E_train_after_infer"] = results.get(f"epoch_{n_epochs}/E_train_after_infer", None)
    results["final/E_test_after_infer"] = results.get(f"epoch_{n_epochs}/E_test_after_infer", None)
    return results


if __name__ == "__main__":
    # 建议你先用这套（接近你之前能到 0.85 的那种“SGD友好推断”）
    cfg = dict(
        seed=0,
        device=None,
        data_root="./data",
        download=True,
        batch_size=256,
        epochs=25,

        layers=[784, 256, 256, 10],
        activation="leaky_relu",
        last_send_activation="identity",

        # 关键：推断别太弱（DiscPC/PC里这两项对 SGD 尤其关键）
        steps_train=8,
        steps_train_max=8,
        steps_train_schedule="constant",
        lr_x=0.03,
        momentum_x=0.9,

        steps_test=100,

        # mini-batch SGD on theta
        lr_theta=1e-3,
        momentum_theta=0.9,
        weight_decay=1e-2,
        grad_clip_norm=0.0,

        # 关键：先 constant，避免“调参时你以为lr是常数，其实在衰减”
        lr_theta_schedule="constant",     # constant|cosine|cosine_warmup
        lr_theta_min=0.0,
        lr_theta_warmup_steps=0,

        label_mode="onehot_01",
        log_every=5,
        num_workers=0,
        pin_memory=False,
        weight_init="xavier",
        user_name="mengfan",
    )
    out = trainable(cfg)
    print("final/test_acc =", out["final/test_acc"])
