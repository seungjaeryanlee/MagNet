import os

import numpy as np

from .download import download_dataset

try:
    import tensorflow as tf
except ImportError:
    pass


def TensorFlowDataset(download_path: str = "data/", download=True):
    """
    Return a TensorFlow-style dataset.

    Parameters
    ----------
    download_path : str
        Path to the downloaded data
    download : bool
        If true, data is downloaded automatically if it does not exist.

    Returns
    -------
    dataset : tf.data.Dataset
        TensorFlow-style dataset
    """
    if download:
        if not os.path.exists(os.path.join(download_path, "dataset.npy")):
            download_dataset(download_path)
    else:
        if not os.path.exists(os.path.join(download_path, "dataset.npy")):
            print("[MagNet] Dataset does not exist. Please call with download=True.")

    datas = np.load(os.path.join(download_path, "dataset.npy"))
    voltage = tf.convert_to_tensor(datas[:, :, 0])
    current = tf.convert_to_tensor(datas[:, :, 1])
    voltage_dataset = tf.data.Dataset.from_tensor_slices(voltage)
    current_dataset = tf.data.Dataset.from_tensor_slices(current)
    dataset = tf.data.Dataset.zip((voltage_dataset, current_dataset))

    return dataset
