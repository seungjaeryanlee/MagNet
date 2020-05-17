"""
This module preprocesses raw data from experiments to dataset format.

Author: Seungjae Ryan Lee
"""
import glob
import re
import time

import numpy as np
from tqdm import tqdm


def preprocess(raw_data_dir="data/raw/", clean_data_dir="data/clean/"):
    data_infos = []
    print(f"Loading raw data from '{raw_data_dir}' to and saving clean data to '{clean_data_dir}'...")
    for filename in tqdm(sorted(glob.glob(f"{raw_data_dir}/I*.csv"))):
        p = re.search("(I|V)\((.*)_(.*)_(.*)\)", filename)
        wave_type = p.group(1)
        wave_shape = p.group(2)
        wave_freq = p.group(3)
        wave_cond = p.group(4)

        # Load CSV pair
        mat_i = np.loadtxt(open(f"{raw_data_dir}/I({wave_shape}_{wave_freq}_{wave_cond}).csv", "rb"), delimiter=",")
        mat_v = np.loadtxt(open(f"{raw_data_dir}/V({wave_shape}_{wave_freq}_{wave_cond}).csv", "rb"), delimiter=",")
        assert mat_i.shape == mat_v.shape, f"Received two CSV files with different shapes - {mat_i.shape} and {mat_v.shape} for I|V({wave_shape}_{wave_freq}_{wave_cond}).csv"
        num_samples, sample_length = mat_v.shape
        # assert sample_length == 8192, f"Expected sample length of 8192 but got {sample_length}"

        # Convert to long list of vectors
        # (num_samples, sample_length) -> (num_samples * sample_length, )
        vecs_i = []
        vecs_v = []
        for i in range(num_samples):
            vec_i = mat_i[i, :]
            vec_v = mat_v[i, :]
            vecs_i.append(vec_i)
            vecs_v.append(vec_v)
        long_vec_i = np.concatenate(vecs_i)
        long_vec_v = np.concatenate(vecs_v)
        assert len(long_vec_i) == num_samples * sample_length
        assert len(long_vec_v) == num_samples * sample_length
        # Stack vectors into a matrix
        long_vec_i = np.expand_dims(long_vec_i, axis=1)
        long_vec_v = np.expand_dims(long_vec_v, axis=1)
        clean_mat = np.concatenate((long_vec_v, long_vec_i), axis=1)

        # Save as CSV
        np.savetxt(
            f"{clean_data_dir}/data_{wave_shape}_{wave_freq}_{wave_cond}.csv",
            clean_mat,
            delimiter=",",
            header=f"2 us,{sample_length}",
            comments='',
        )
        data_info = [f"data_{wave_shape}_{wave_freq}_{wave_cond}.csv", wave_shape, wave_freq, wave_cond, num_samples, "2 us", sample_length]
        data_infos.append(data_info)

    data_infos = np.vstack(data_infos)
    np.savetxt(
        f"{clean_data_dir}/info.csv",
        data_infos,
        delimiter=",",
        header="filename,shape,frequency,condition,num_samples,sample_period,sample_length",
        comments="",
        fmt="%s",
    )

if __name__ == "__main__":
    start_time = time.time()
    preprocess()
    end_time = time.time()
    print(f"This preprocessing took {end_time - start_time} seconds.")
