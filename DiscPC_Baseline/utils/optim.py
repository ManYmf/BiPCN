# utils/optim.py
import torch


class AdamW:
    """
    Minimal manual AdamW for local grads.
    NOTE: grads must be aligned with params passed in.
    """

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
