import glob
import os

import numpy as np
import pandas as pd
import torch


def get_pytorch_dataset(data_path: str = "data/"):
    """
    Return a PyTorch-style dataset.

    Parameters
    ----------
    data_path : str
        Path to the downloaded data

    Returns
    -------
    dataset : torch.utils.data.Dataset
        PyTorch-style dataset
    """
    csv_paths = glob.glob(os.path.join(data_path, "clean/*.csv"))

    # TODO(seungjaeryanlee): Probably better to do this during download_dataset()
    datas = []
    for csv_path in csv_paths:
        # Ignore the meta-info CSV
        if "info.csv" in csv_path:
            continue
        df = pd.read_csv(csv_path, header=None)
        sample_period, sample_length = df.iloc[0]
        # TODO(seungjaeryanlee): Only support data with sample length of 8192
        if sample_length != 8192:
            continue
        data = df.iloc[1:].values.astype(np.float64)
        data = np.split(data, int(data.shape[0] / sample_length))
        data = np.array(data)
        datas.append(data)
    datas = np.concatenate(datas)

    voltage = torch.FloatTensor(datas[:, :, 0])
    current = torch.FloatTensor(datas[:, :, 1])
    dataset = torch.utils.data.TensorDataset(voltage, current)

    return dataset
