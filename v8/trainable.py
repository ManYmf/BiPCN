def trainable(config: dict):
    """
    Pure-function style trainable(config) -> results dict
    - config: hyperparams & runtime options
    - returns: metrics dict (epoch-wise + final)
    """
    results = {}

    import torch
    import numpy as np
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # --------------------------
    # 0. Config with defaults
    # --------------------------
    seed = int(config.get("seed", 0))
    batch_size = int(config.get("batch_size", 64))
    n_epochs = int(config.get("n_epochs", 50))
    layers = config.get("layers", [784, 128, 64, 10])

    m = int(config.get("m", 15))
    lr_activity = float(config.get("lr_activity", 0.03))
    lr_weight = float(config.get("lr_weight", 0.03))
    lambda_target = float(config.get("lambda_target", 0.7))
    use_bias = bool(config.get("use_bias", True))

    # device: allow override, otherwise auto
    device_cfg = config.get("device", None)
    if device_cfg is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    results["device"] = str(device)
    results["torch.cuda.is_available()"] = bool(torch.cuda.is_available())

    # reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --------------------------
    # 1. Activations
    # --------------------------
    def sigmoid(x):
        return torch.sigmoid(torch.clamp(x, -20, 20))

    def softmax_stable(x, dim=0):
        # stable softmax for shape (C, B) over class dim=0
        x = torch.clamp(x, -20, 20)
        x = x - torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x)
        return exp_x / (torch.sum(exp_x, dim=dim, keepdim=True) + 1e-12)

    # --------------------------
    # 2. Data loading
    # --------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=False)

    results["train_batches"] = len(train_loader)
    results["test_batches"] = len(test_loader)
    results["layers"] = list(layers)

    # --------------------------
    # 3. discPC network (same logic, wrapped)
    # --------------------------
    class discPC_Flexible:
        def __init__(self, layers, activations=None, use_bias=True, device="cpu"):
            self.layers = layers
            self.n_layers = len(layers)
            self.use_bias = use_bias
            self.device = device

            if activations is None:
                self.activations = [sigmoid] * (self.n_layers - 2) + [lambda z: softmax_stable(z, dim=0)]
            else:
                self.activations = activations

            self.weights = []
            self.biases = []
            for i in range(self.n_layers - 1):
                input_dim = layers[i]
                output_dim = layers[i + 1]
                w = torch.randn(input_dim, output_dim, device=device) * torch.sqrt(
                    torch.tensor(1.0 / input_dim, device=device)
                )
                self.weights.append(w)
                if use_bias:
                    self.biases.append(torch.zeros(output_dim, 1, device=device))
                else:
                    self.biases.append(None)

        def initialize_activity(self, x_batch):
            activities = [x_batch]
            for i in range(self.n_layers - 1):
                prev = activities[-1]
                z = self.weights[i].T @ prev
                if self.use_bias and self.biases[i] is not None:
                    z = z + self.biases[i]
                a = self.activations[i](z)
                activities.append(a)
            return activities

        def update_activity(self, x_batch, activities, target=None, lr=0.1, m=20, lambda_target=0.9):
            current = [a.clone() for a in activities]
            for _ in range(m):
                # forward prediction from x_batch
                pred = [x_batch]
                for i in range(self.n_layers - 1):
                    prev = pred[-1]
                    z = self.weights[i].T @ prev
                    if self.use_bias and self.biases[i] is not None:
                        z = z + self.biases[i]
                    pred.append(self.activations[i](z))

                # update activities (reverse)
                for i in reversed(range(1, self.n_layers)):
                    eps = current[i] - pred[i]
                    update = -lr * eps

                    if target is not None and i == self.n_layers - 1:
                        # output target clamp
                        tgt_err = current[i] - target
                        update = update - lr * lambda_target * tgt_err

                    elif target is not None and i < self.n_layers - 1:
                        # hidden target influence (same structure as your original)
                        sig_deriv = pred[i] * (1.0 - pred[i])
                        if i + 1 == self.n_layers - 1:
                            next_err = current[i + 1] - target
                        else:
                            next_err = current[i + 1] - pred[i + 1]
                        tgt_err = (self.weights[i] @ next_err) * sig_deriv
                        update = update - lr * lambda_target * tgt_err

                    current[i] = current[i] + update
            return current

        def update_weight(self, x_batch, activities, target, lr=0.01):
            # forward prediction from x_batch
            pred = [x_batch]
            for i in range(self.n_layers - 1):
                prev = pred[-1]
                z = self.weights[i].T @ prev
                if self.use_bias and self.biases[i] is not None:
                    z = z + self.biases[i]
                pred.append(self.activations[i](z))

            # errors (same as your original)
            errors = [None] * (self.n_layers - 1)
            errors[-1] = target - pred[-1]
            for i in reversed(range(self.n_layers - 2)):
                sig_deriv = pred[i + 1] * (1.0 - pred[i + 1])
                errors[i] = (self.weights[i + 1] @ errors[i + 1]) * sig_deriv

            B = x_batch.shape[1]
            for i in range(self.n_layers - 1):
                self.weights[i] = self.weights[i] + lr * (pred[i] @ errors[i].T) / max(B, 1)
                if self.use_bias and self.biases[i] is not None:
                    self.biases[i] = self.biases[i] + lr * torch.mean(errors[i], dim=1, keepdim=True)

        def predict(self, x_batch, m=20):
            acts = self.initialize_activity(x_batch)
            acts = self.update_activity(x_batch, acts, m=m)
            return torch.argmax(acts[-1], dim=0)

    model = discPC_Flexible(layers=layers, use_bias=use_bias, device=device)

    # --------------------------
    # 4. Train / Eval loops
    # --------------------------
    for epoch in range(n_epochs):
        model_train_loss = 0.0
        correct = 0
        total = 0

        for data, targets in train_loader:
            # x: (784, B)
            x_batch = data.view(data.size(0), -1).T.to(device, dtype=torch.float32)
            # y one-hot: (10, B)
            y_idx = targets.to(device)
            y_onehot = F.one_hot(y_idx, num_classes=10).to(device, dtype=torch.float32).T

            acts = model.initialize_activity(x_batch)
            acts = model.update_activity(
                x_batch, acts, target=y_onehot, lr=lr_activity, m=m, lambda_target=lambda_target
            )
            model.update_weight(x_batch, acts, y_onehot, lr=lr_weight)

            y_pred = acts[-1]
            batch_loss = torch.mean((y_pred - y_onehot) ** 2)
            model_train_loss += float(batch_loss.item())

            pred_labels = torch.argmax(y_pred, dim=0)
            correct += int((pred_labels == y_idx).sum().item())
            total += int(y_idx.numel())

        train_loss = model_train_loss / max(len(train_loader), 1)
        train_acc = correct / max(total, 1)

        # test
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                x_batch = data.view(data.size(0), -1).T.to(device, dtype=torch.float32)
                y_idx = targets.to(device)

                acts = model.initialize_activity(x_batch)
                acts = model.update_activity(x_batch, acts, m=m)
                pred_labels = torch.argmax(acts[-1], dim=0)

                test_correct += int((pred_labels == y_idx).sum().item())
                test_total += int(y_idx.numel())

        test_acc = test_correct / max(test_total, 1)

        # log into results dict
        results[f"epoch_{epoch+1}/train_loss"] = train_loss
        results[f"epoch_{epoch+1}/train_acc"] = train_acc
        results[f"epoch_{epoch+1}/test_acc"] = test_acc

    # --------------------------
    # 5. Final per-class accuracy (optional but useful)
    # --------------------------
    class_correct = torch.zeros(10, device=device, dtype=torch.long)
    class_total = torch.zeros(10, device=device, dtype=torch.long)

    with torch.no_grad():
        for data, targets in test_loader:
            x_batch = data.view(data.size(0), -1).T.to(device, dtype=torch.float32)
            y_idx = targets.to(device)

            acts = model.initialize_activity(x_batch)
            acts = model.update_activity(x_batch, acts, m=m)
            pred = torch.argmax(acts[-1], dim=0)

            for k in range(10):
                mask = (y_idx == k)
                class_total[k] += mask.sum()
                class_correct[k] += (pred[mask] == k).sum()

    per_class_acc = (class_correct.float() / (class_total.float() + 1e-12)).detach().cpu().tolist()
    results["final/test_acc"] = results[f"epoch_{n_epochs}/test_acc"]
    results["final/train_acc"] = results[f"epoch_{n_epochs}/train_acc"]
    results["final/train_loss"] = results[f"epoch_{n_epochs}/train_loss"]
    results["final/per_class_acc"] = per_class_acc

    return results
