import os
import tarfile

import gdown


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

    url = "https://drive.google.com/uc?id=1fUfSH3odWmr5ORc3iZk3hPNbRr61f4Ls"
    zipped_file_path = os.path.join(download_path, "magnet.tar.gz")
    gdown.download(url, zipped_file_path, quiet=False)
    print("[MagNet] Dataset downloaded successfully.")

    with tarfile.open(zipped_file_path, "r:gz") as tar:
        tar.extractall(download_path)
    print("[MagNet] Dataset unzipped successfully.")

    os.remove(zipped_file_path)
    print("[MagNet] Cleanup successful.")

    print("███╗   ███╗ █████╗  ██████╗ ███╗   ██╗███████╗████████╗")
    print("████╗ ████║██╔══██╗██╔════╝ ████╗  ██║██╔════╝╚══██╔══╝")
    print("██╔████╔██║███████║██║  ███╗██╔██╗ ██║█████╗     ██║   ")
    print("██║╚██╔╝██║██╔══██║██║   ██║██║╚██╗██║██╔══╝     ██║   ")
    print("██║ ╚═╝ ██║██║  ██║╚██████╔╝██║ ╚████║███████╗   ██║   ")
    print("╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ")
