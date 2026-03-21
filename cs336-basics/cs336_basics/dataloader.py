import numpy as np
import torch
import numpy.typing as npt
import os
from typing import BinaryIO, IO


def load_dataset_mmap(path: str, dtype: np.dtype = np.uint16) -> npt.NDArray:
    """
    Memory-map a .npy token array from disk.  The array is *not* loaded into
    RAM; pages are fetched lazily by the OS when accessed.

    Args:
        path:  Path to a .npy file saved with np.save.
        dtype: Must match the dtype used when the file was created.

    Returns:
        A read-only numpy array backed by the file on disk.
    """
    data = np.load(path, mmap_mode="r")
    # Ensure the dtype matches what we expect (avoids silent reinterpretation)
    if data.dtype != np.dtype(dtype):
        raise ValueError(
            f"dtype mismatch: file contains {data.dtype}, expected {dtype}. "
            "Pass the correct dtype or re-create the dataset."
        )
    return data


def get_datapoints_from_source(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    max_start = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    x = torch.stack([torch.from_numpy(dataset[s : s + context_length].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(dataset[s + 1 : s + context_length + 1].astype(np.int64)) for s in starts])
    return x.to(device), y.to(device) 

def save_checkpoint(model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    wandb_run_id: str | None = None,):

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'wandb_run_id': wandb_run_id,
    }, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,):

    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    wandb_run_id = checkpoint.get('wandb_run_id', None)
    return checkpoint['iteration'], wandb_run_id
    
