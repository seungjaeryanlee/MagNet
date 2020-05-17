# Preprocessing Dataset

The raw data should be added to `data/raw/` directory. Then, run `preprocess.py` to preprocess them to dataset format.

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
