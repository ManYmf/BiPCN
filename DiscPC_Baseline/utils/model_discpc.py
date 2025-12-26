# utils/models_discpc.py
import math
import torch
from utils.activations import get_activation


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
        self.L = len(self.layers) - 1  # number of weight matrices
        self.use_bias = bool(use_bias)
        self.device = torch.device(device)

        # sending activations f(x_l) used in prediction to next layer
        f, fprime = get_activation(activation, negative_slope=negative_slope)
        f_last, fprime_last = get_activation(last_send_activation, negative_slope=negative_slope)

        self.send_f = [f for _ in range(self.L)]
        self.send_fprime = [fprime for _ in range(self.L)]
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

    @torch.no_grad()
    def update_theta_local(self, acts):
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
