import os
import torch


def get_device() -> torch.device:
    """Select compute device.

    For SNN simulations with small/medium tensors and heavy conditional logic,
    CPU is typically faster than MPS due to GPU kernel launch overhead.
    MPS/CUDA become beneficial only with large networks (10k+ neurons, 1M+ synapses).

    Override with BRAIN_DEVICE=mps or BRAIN_DEVICE=cuda environment variable.
    """
    override = os.environ.get("BRAIN_DEVICE", "").lower()
    if override:
        return torch.device(override)

    return torch.device("cpu")


DEVICE = get_device()
