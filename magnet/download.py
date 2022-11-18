import glob
import os
import tarfile

import gdown
import numpy as np
import pandas as pd


def download_dataset(download_path: str = "data/"):
    """
    Download dataset from Google Drive

    The dataset is a `magnet.tar.gz` file that is unzipped upon download. The
    dataset only contains "clean", formatted data.

    Parameters
    ----------
    download_path : str
        The path to the folder to download the dataset to.
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    url = "https://drive.google.com/uc?id=1OJyehwedyFBfvNJUavvIrhckiYXcTo0Q"
    zipped_file_path = os.path.join(download_path, "20201118.tar.gz")
    gdown.download(url, zipped_file_path, quiet=False)
    print("[MagNet] Dataset downloaded successfully.")

    with tarfile.open(zipped_file_path, "r:gz") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, download_path)
    print("[MagNet] Dataset unzipped successfully.")

    os.remove(zipped_file_path)
    print("[MagNet] Cleanup successful.")

    print("[MagNet] Preprocessing dataset. This may take a few minutes.")
    csv_paths = glob.glob(os.path.join(download_path, "dataset/*.csv"))
    datas = []
    for csv_path in csv_paths:
        # Ignore the meta-info CSV
        if "info.csv" in csv_path:
            continue

        df = pd.read_csv(csv_path, header=None)
        sample_period, sample_length = df.iloc[0]

        # TODO(seungjaeryanlee): Support data with different sample length (by zero padding?)
        # Currently, data_trap_1k, 2k, 5k have sample length 8000 instead of 8192
        if sample_length != 8192:
            print(f"[MagNet] WARNING: {csv_path} has sample length {sample_length}, which is not the expected 8192.")
            continue

        data = df.iloc[1:].values.astype(np.float64)
        data = np.split(data, int(data.shape[0] / sample_length))
        data = np.array(data)
        datas.append(data)
    datas = np.concatenate(datas)
    np.save(os.path.join(download_path, "dataset.npy"), datas)
    print("[MagNet] Preprocessing finished successfully.")

    print("███╗   ███╗ █████╗  ██████╗ ███╗   ██╗███████╗████████╗")
    print("████╗ ████║██╔══██╗██╔════╝ ████╗  ██║██╔════╝╚══██╔══╝")
    print("██╔████╔██║███████║██║  ███╗██╔██╗ ██║█████╗     ██║   ")
    print("██║╚██╔╝██║██╔══██║██║   ██║██║╚██╗██║██╔══╝     ██║   ")
    print("██║ ╚═╝ ██║██║  ██║╚██████╔╝██║ ╚████║███████╗   ██║   ")
    print("╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ")
