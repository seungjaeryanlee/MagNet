import os

import numpy as np

from .download import download_dataset

try:
    import torch
except ImportError:
    pass


def PyTorchDataset(download_path: str = "data/", download=True):
    """
    Return a PyTorch-style dataset.

    Parameters
    ----------
    download_path : str
        Path to the downloaded data
    download : bool
        If true, data is downloaded automatically if it does not exist.

    Returns
    -------
    dataset : torch.utils.data.Dataset
        PyTorch-style dataset
    """
    if download:
        if not os.path.exists(os.path.join(download_path, "dataset.npy")):
            download_dataset(download_path)
    else:
        if not os.path.exists(os.path.join(download_path, "dataset.npy")):
            print("[MagNet] Dataset does not exist. Please call with download=True.")

    datas = np.load(os.path.join(download_path, "dataset.npy"))
    voltage = torch.FloatTensor(datas[:, :, 0])
    current = torch.FloatTensor(datas[:, :, 1])
    dataset = torch.utils.data.TensorDataset(voltage, current)

    return dataset
