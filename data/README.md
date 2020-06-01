# Dataset

Due to [GitHub size limitations](https://help.github.com/en/github/managing-large-files/what-is-my-disk-quota), this repository does not contain the dataset itself. Instead, we provide the dataset through Google Drive, which you can download using a web browser or `gdown`. Alternatively, you can download the raw data and run the preprocessing script in this folder.

```
📂 raw                # Raw Data collected
📂 clean              # Data formatted using `preprocess.py`
📃 preprocess.py      # Script for preprocessing and reformatting data
```

## Download Clean Data

You can download the data in one zipped file or individually using Google Drive UI.

- [magnet.zip](https://drive.google.com/file/d/1iQFNT2kz_0pfyvU3uTnp6tEgiYSFVF5r/view?usp=sharing)
- [magnet.tar.gz](https://drive.google.com/file/d/1fUfSH3odWmr5ORc3iZk3hPNbRr61f4Ls/view?usp=sharing)
- [Individual Files](https://drive.google.com/drive/folders/1lCijIadT-JxrgzMz_bec1buFVpGJQmly?usp=sharing)

Alternatively, you can use the `gdown` package. Download the package using `pip`.

```
pip install gdown
```

Then, you can use `gdown` in terminal to download any of the zipped files.

```
# For magnet.zip
gdown https://drive.google.com/uc?id=1iQFNT2kz_0pfyvU3uTnp6tEgiYSFVF5r
# For magnet.tar.gzd
gdown https://drive.google.com/uc?id=1fUfSH3odWmr5ORc3iZk3hPNbRr61f4Ls
```

## Preprocessing Raw Data to Clean Data

You can also download the raw data and run the preprocessing script to generate the dataset. The raw data should be added to `data/raw/` directory. Then, go to the `data/` directory and run `preprocess.py` to preprocess them to dataset format.

```
python preprocess.py
```

The processed "clean" data will be generated in `data/clean/` directory. You can zip the directory by running the following commands in the `data/` directory:

```
# For .tar.gz
tar -czvf magnet.tar.gz clean/
# For .zip
zip -r magnet.zip clean/
```

For each pair of CSV files with names `V(<shape>_<frequency>_<condition>).csv` and `I(<shape>_<frequency>_<condition>).csv`, a single CSV file `data_<shape>_<frequency>_<condition>.csv` is generated. For example, `V(tri_1k_hiB).csv` and `I(tri_1k_hiB).csv` is processed to generate `data_tri_1k_hiB.csv`.
