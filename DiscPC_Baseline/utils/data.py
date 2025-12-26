# utils/data.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import radas


def load_mnist_data(
    batch_size=256,
    data_root=None,          # None -> radas.get_data_dir(user_name)/data
    download=False,
    num_workers=0,
    pin_memory=False,
    user_name="mengfan",
):
    """
    MNIST loader scaled to [-1, 1].
    If data_root is None, use:
        radas.get_data_dir(user_name)/data
    """
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
