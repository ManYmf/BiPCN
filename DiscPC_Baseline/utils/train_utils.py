# utils/train_utils.py
import numpy as np
import torch


def set_seed(seed: int):
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic_cudnn(deterministic: bool = True):
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False


def get_device(device_cfg=None) -> torch.device:
    if device_cfg is not None:
        return torch.device(device_cfg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
