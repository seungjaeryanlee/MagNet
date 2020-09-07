import os

import gdown


def download_dataset(download_path="data/"):
    os.makedirs(download_path)

    url = "https://drive.google.com/uc?id=1fUfSH3odWmr5ORc3iZk3hPNbRr61f4Ls"
    output = os.path.join(download_path, "magnet.tar.gz")
    gdown.download(url, output, quiet=False)
