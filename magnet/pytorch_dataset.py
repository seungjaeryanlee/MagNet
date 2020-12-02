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


# TODO(seungjaeryanlee): Find a better name
def PyTorchVoltageToCoreLossDataset(download_path: str = "data/", download=True):
    """
    Return a PyTorch-style dataset for predicting core loss from voltage data.

    Parameters
    ----------
    download_path : str
        Path to the downloaded data
    download : bool
        If true, data is downloaded automatically if it does not exist.

    Returns
    -------
    dataset : torch.utils.data.Dataset
        PyTorch-style dataset for predicting core loss from voltage data
    """
    if download:
        if not os.path.exists(os.path.join(download_path, "dataset.npy")):
            download_dataset(download_path)
    else:
        if not os.path.exists(os.path.join(download_path, "dataset.npy")):
            print("[MagNet] Dataset does not exist. Please call with download=True.")

    # TODO(seungjaeryanlee): Add these as parameters to method
    DATA_LENGTH = 400
    SAMPLE_RATE = 2e-6

    datas = np.load(os.path.join(download_path, "dataset.npy"))
    voltage = torch.FloatTensor(datas[:, :DATA_LENGTH, 0])
    current = torch.FloatTensor(datas[:, :DATA_LENGTH, 1])

    power = voltage * current
    t = np.arange(0, (DATA_LENGTH - 0.5) * SAMPLE_RATE, SAMPLE_RATE)
    core_loss = np.trapz(power, t, axis=1) / (SAMPLE_RATE * DATA_LENGTH)
    core_loss = torch.FloatTensor(core_loss).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(voltage, core_loss)

    return dataset
