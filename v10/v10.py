# v5_MNIST_discpc_trainable.py
# DiscPC (as in arXiv:2505.23415v1) for MNIST classification
#
# Key alignment to paper:
# - Energy: E_disc = sum_{l=2..L} 1/2 || x_l - V_{l-1} f(x_{l-1}) ||^2
# - Inference dynamics (disc-only special case of Eq.(5)):
#     dx_l/dt = -eps_l + f'(x_l) ⊙ (V_l^T eps_{l+1})   for l=2..L-1 (paper indexing)
#   with eps_l := x_l - V_{l-1} f(x_{l-1})
# - Weight update (Eq.(7)):
#     ΔV_l ∝ eps_{l+1} f(x_l)^T   (Hebbian/local outer product)
#
# Implementation notes (0-indexed layers):
# - activities[0] == x1 (image), activities[L] == xL (label/logits)
# - weights[l] has shape (dim_l, dim_{l+1}) and prediction is:
#       pred[l+1] = weights[l].T @ f(activities[l]) + bias[l]
#
# Train: clamp activities[0] and activities[L] (image + one-hot label)
# Test: clamp activities[0] only; infer activities; predict argmax(activities[L])

import os
import math
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --------------------------
# Activations for "sending" f(x_l) used in predictions V_l f(x_l)
# --------------------------
def _identity(x):  # noqa: D401
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
    # Exact derivative of GELU (approx used by PyTorch is close; we use exact for stability)
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
    data_root="./data",
    download=True,
    num_workers=0,
    pin_memory=False,
):
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
# Minimal AdamW (manual) for local gradients
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
            self.state.append(
                dict(
                    t=0,
                    m=torch.zeros_like(p),
                    v=torch.zeros_like(p),
                )
            )

    @torch.no_grad()
    def step(self, grads):
        for p, g, st in zip(self.params, grads, self.state):
            st["t"] += 1
            t = st["t"]
            st["m"].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            st["v"].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

            m_hat = st["m"] / (1.0 - (self.beta1**t))
            v_hat = st["v"] / (1.0 - (self.beta2**t))

            # decoupled weight decay
            if self.weight_decay > 0:
                p.mul_(1.0 - self.lr * self.weight_decay)

            p.addcdiv_(m_hat, torch.sqrt(v_hat).add_(self.eps), value=-self.lr)


# --------------------------
# DiscPC MLP (bottom-up only)
# --------------------------
class DiscPCMLP:
    def __init__(
        self,
        layers,
        activation="leaky_relu",
        last_send_activation="identity",  # per paper, last disc projection often uses identity
        use_bias=True,
        device="cpu",
        negative_slope=0.01,
        weight_init="xavier",
    ):
        self.layers = list(layers)
        self.L = len(self.layers) - 1  # number of weight matrices
        self.use_bias = bool(use_bias)
        self.device = torch.device(device)

        # sending activations f(x_l) used in prediction to next layer
        f, fprime = get_activation(activation, negative_slope=negative_slope)
        f_last, fprime_last = get_activation(last_send_activation, negative_slope=negative_slope)

        self.send_f = [f for _ in range(self.L)]
        self.send_fprime = [fprime for _ in range(self.L)]
        # override last sending activation (from last hidden to output)
        self.send_f[-1] = f_last
        self.send_fprime[-1] = fprime_last

        # parameters
        self.weights = []
        self.biases = []
        for l in range(self.L):
            din = self.layers[l]
            dout = self.layers[l + 1]
            W = torch.empty(din, dout, device=self.device, dtype=torch.float32)

            if weight_init == "xavier":
                # Xavier uniform for stability
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

        # will be filled in trainable (optimizer choice)
        self.opt_theta = None

    def _predict_next(self, l, x_l):
        # prediction for layer l+1 from x_l
        z = self.weights[l].T @ self.send_f[l](x_l)
        if self.use_bias and (self.biases[l] is not None):
            z = z + self.biases[l]
        return z

    def init_bottom_up(self, x0, clamp_output=False, xL=None):
        # activities[0]=x0; init hidden (and optionally output) by bottom-up sweep
        acts = [None] * (self.L + 1)
        acts[0] = x0

        # init 1..L-1 by sweep
        for l in range(self.L - 1):
            acts[l + 1] = self._predict_next(l, acts[l])

        # output
        if clamp_output:
            if xL is None:
                raise ValueError("clamp_output=True but xL is None")
            acts[self.L] = xL
        else:
            acts[self.L] = self._predict_next(self.L - 1, acts[self.L - 1])

        return acts

    def preds_and_eps(self, acts):
        # preds[k] predicts acts[k], for k>=1. eps[k] = acts[k] - preds[k]
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
        # SGD + momentum on activities (paper: activity updates use SGD w/ momentum)
        vel = [None] * (self.L + 1)
        for k in range(1, self.L + 1):
            vel[k] = torch.zeros_like(acts[k])

        for _ in range(int(steps)):
            _, eps = self.preds_and_eps(acts)

            # update hidden layers 1..L-1 always (if not clamped by design)
            for l in range(1, self.L):
                # discPC dynamics: dx_l = -eps[l] + f'(x_l) ⊙ (W_l @ eps[l+1])
                dx = -eps[l] + self.send_fprime[l](acts[l]) * (self.weights[l] @ eps[l + 1])
                vel[l].mul_(momentum_x).add_(dx)
                acts[l].add_(vel[l], alpha=lr_x)

            # optionally update output if not clamped (dx_L = -eps[L])
            if not clamp_output:
                dxL = -eps[self.L]
                vel[self.L].mul_(momentum_x).add_(dxL)
                acts[self.L].add_(vel[self.L], alpha=lr_x)

            # re-clamp
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
        # E_disc = sum_{k=1..L} 1/2 || eps[k] ||^2  (batch-mean)
        _, eps = self.preds_and_eps(acts)
        B = acts[0].shape[1]
        e = 0.0
        for k in range(1, self.L + 1):
            e += 0.5 * float((eps[k] * eps[k]).sum().item()) / max(1, B)
        return e

    @torch.no_grad()
    def update_theta_local(self, acts):
        # Eq.(7) local parameter update using eps and f(x_l)
        _, eps = self.preds_and_eps(acts)
        B = acts[0].shape[1]

        grads = []  # grads aligned to [W0,b0,W1,b1,...] if using AdamW
        for l in range(self.L):
            # dE/dW = - f(x_l) @ eps[l+1]^T / B   because pred = W^T f(x)
            fx = self.send_f[l](acts[l])
            gradW = -(fx @ eps[l + 1].T) / max(1, B)
            grads.append(gradW)

            if self.use_bias and (self.biases[l] is not None):
                gradb = -eps[l + 1].mean(dim=1, keepdim=True)
                grads.append(gradb)

        if self.opt_theta is None:
            raise RuntimeError("opt_theta is not set")
        self.opt_theta.step(grads)

    @torch.no_grad()
    def predict_class(self, x0, steps_eval=100, lr_x=0.01, momentum_x=0.0):
        # classification procedure in paper: clamp x1, bottom-up init, 100 inference steps, argmax(xL)
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
        logits = acts[self.L]  # (10, B)
        return torch.argmax(logits, dim=0)


# --------------------------
# trainable(config): single entry point
# --------------------------
def trainable(config: dict):
    results = {}

    # reproducibility
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # device
    device_cfg = config.get("device", None)
    device = torch.device(device_cfg) if device_cfg is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results["device"] = str(device)
    results["torch.cuda.is_available"] = bool(torch.cuda.is_available())

    # data
    batch_size = int(config.get("batch_size", 256))
    data_root = config.get("data_root", "./data")
    download = bool(config.get("download", True))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", False))

    train_loader, test_loader = load_mnist_data(
        batch_size=batch_size,
        data_root=data_root,
        download=download,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # model hyperparams (paper-like defaults)
    layers = config.get("layers", [784, 256, 256, 10])
    activation = config.get("activation", "leaky_relu")
    last_send_activation = config.get("last_send_activation", "identity")
    negative_slope = float(config.get("negative_slope", 0.01))
    use_bias = bool(config.get("use_bias", True))
    weight_init = config.get("weight_init", "xavier")

    steps_train = int(config.get("steps_train", 8))     # T
    steps_test = int(config.get("steps_test", 100))     # T_eval
    lr_x = float(config.get("lr_x", 0.01))
    momentum_x = float(config.get("momentum_x", 0.0))

    # theta optimiser (paper uses AdamW)
    lr_theta = float(config.get("lr_theta", 1e-4))
    weight_decay = float(config.get("weight_decay", 5e-3))
    adam_beta1 = float(config.get("adam_beta1", 0.9))
    adam_beta2 = float(config.get("adam_beta2", 0.999))
    adam_eps = float(config.get("adam_eps", 1e-8))

    n_epochs = int(config.get("epochs", 25))
    log_every = int(config.get("log_every", 5))

    # label encoding
    label_mode = str(config.get("label_mode", "onehot_01")).lower()  # "onehot_01" or "onehot_pm1"

    # record config (for downstream df aggregation)
    results["config/seed"] = seed
    results["config/batch_size"] = batch_size
    results["config/epochs"] = n_epochs
    results["config/layers"] = str(layers)
    results["config/activation"] = str(activation)
    results["config/last_send_activation"] = str(last_send_activation)
    results["config/steps_train"] = steps_train
    results["config/steps_test"] = steps_test
    results["config/lr_x"] = lr_x
    results["config/momentum_x"] = momentum_x
    results["config/lr_theta"] = lr_theta
    results["config/weight_decay"] = weight_decay
    results["config/label_mode"] = label_mode

    # build model
    model = DiscPCMLP(
        layers=layers,
        activation=activation,
        last_send_activation=last_send_activation,
        use_bias=use_bias,
        device=device,
        negative_slope=negative_slope,
        weight_init=weight_init,
    )

    # attach AdamW over [W0,b0,W1,b1,...]
    theta_params = []
    for l in range(model.L):
        theta_params.append(model.weights[l])
        if model.use_bias and (model.biases[l] is not None):
            theta_params.append(model.biases[l])
    model.opt_theta = AdamW(
        theta_params,
        lr=lr_theta,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
        weight_decay=weight_decay,
    )

    # helper: make label xL (10,B)
    def make_label_onehot(targets: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.one_hot(targets, num_classes=10).to(torch.float32)  # (B,10)
        if label_mode == "onehot_pm1":
            y = 2.0 * y - 1.0
        return y.T  # (10,B)

    # training loop
    for epoch in range(n_epochs):
        model_energy_sum = 0.0
        n_batches = 0

        for data, targets in train_loader:
            # x0: (784,B)
            x0 = data.view(data.size(0), -1).to(device=device, dtype=torch.float32).T
            y = make_label_onehot(targets.to(device=device))

            # init: clamp x0 and xL, bottom-up sweep for hidden
            acts = model.init_bottom_up(x0, clamp_output=True, xL=y)

            # inference (K steps) with clamped ends
            acts = model.infer(
                acts,
                steps=steps_train,
                lr_x=lr_x,
                momentum_x=momentum_x,
                clamp_input=True,
                x0=x0,
                clamp_output=True,
                xL=y,
            )

            # local parameter update (Eq.7)
            model.update_theta_local(acts)

            # track (clamped) energy after inference
            model_energy_sum += model.energy(acts)
            n_batches += 1

        E_train = model_energy_sum / max(1, n_batches)

        # evaluation: classification accuracy (paper procedure)
        test_correct = 0
        test_total = 0

        # also log "clamped energy" on test set (image+true label clamped) after inference
        E_test_sum = 0.0
        E_test_batches = 0

        with torch.no_grad():
            for data, targets in test_loader:
                x0 = data.view(data.size(0), -1).to(device=device, dtype=torch.float32).T
                targets_dev = targets.to(device=device)
                y = make_label_onehot(targets_dev)

                # accuracy: clamp x0 only
                pred = model.predict_class(
                    x0,
                    steps_eval=steps_test,
                    lr_x=lr_x,
                    momentum_x=momentum_x,
                )
                test_correct += int((pred == targets_dev).sum().item())
                test_total += int(targets_dev.numel())

                # energy: clamp x0 + true y (useful for monitoring convergence/energy scale)
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
        results[f"epoch_{ep}/E_train_after_infer"] = float(E_train)
        results[f"epoch_{ep}/test_acc"] = float(test_acc)
        results[f"epoch_{ep}/E_test_after_infer"] = float(E_test)

        if log_every > 0 and (ep % log_every == 0 or ep == 1 or ep == n_epochs):
            print(
                f"Epoch {ep:3d} | E_train={E_train:.6f} | "
                f"test_acc={test_acc:.4f} | E_test(clamped)={E_test:.6f}"
            )

    results["final/test_acc"] = results.get(f"epoch_{n_epochs}/test_acc", None)
    results["final/E_train_after_infer"] = results.get(f"epoch_{n_epochs}/E_train_after_infer", None)
    results["final/E_test_after_infer"] = results.get(f"epoch_{n_epochs}/E_test_after_infer", None)

    return results


if __name__ == "__main__":
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
        steps_train=8,
        steps_test=100,
        lr_x=0.01,
        momentum_x=0.0,
        lr_theta=1e-4,
        weight_decay=5e-3,
        label_mode="onehot_01",
        log_every=5,
        num_workers=0,
        pin_memory=False,
        weight_init="xavier",
    )
    out = trainable(cfg)
    print("final/test_acc =", out["final/test_acc"])
