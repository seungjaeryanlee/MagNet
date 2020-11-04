# MagNet

![PyPI](https://img.shields.io/pypi/v/mag-net?color=blue)
![wandb](https://img.shields.io/badge/wandb-metric-yellow)

MagNet is a large-scale dataset designed to enable researchers modeling magnetic core loss using machine learning to accelerate the design process of power electronics. The dataset contains a large amount of voltage and current data of different magnetic components with different shapes of waveforms and different properties measured in the real world. Researchers may use these data as pairs of excitations and responses to build up dynamic magnetic models or calculate the core loss to derive static models.



## Installation

The trained models are provided via the `mag-net` PyPI package.

```
pip install mag-net
```



## How to Use

### Use Trained Model

The `mag-net` package will soon support download and use of trained models. Please look forward to it!

### Use Dataset

To use the dataset to train a model yourself, you must first download the dataset. The following code will download the dataset to `data/` directory.

```python
import magnet

magnet.download_dataset(download_path="data/")
```

The `mag-net` package will soon support PyTorch-style and TensorFlow-style datasets. Please look forward to it!



## How to Cite

If you used MagNet, please cite us with the following BibTeX item.

<!-- TODO: Update once COMPEL 2020 happens -->

```
@INPROCEEDINGS{MagNet,
  author={H. {Li} and S. {Lee} and M. {Luo} and C. {Sullivan} and Y. {Chen} and M. {Chen}},
  booktitle={2020 Twenty-first IEEE Workshop on Control and Modeling for Power Electronics (COMPEL)}, 
  title={MagNet: A Machine Learning Framework for Magnetic Core Loss Modeling}, 
  year={2020}
}
```
