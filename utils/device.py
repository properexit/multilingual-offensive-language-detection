import torch


def get_device(prefer_gpu=True, force_device=None):
    """
    Returns the best available device.
    Priority:
    1. force_device (if provided)
    2. CUDA
    3. MPS (Apple Silicon)
    4. CPU
    """

    if force_device is not None:
        return torch.device(force_device)

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")