# utils/activations.py
import math
import torch


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
