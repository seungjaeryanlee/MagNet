![MagNet Logo](images/magnet_logo.jpg)

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

The `mag-net` package has **PyTorch** pretrained models that you can load and use.

```python
import magnet

magnet.models.pytorch.MiniLSTM(pretrained=True)
```

If you want to train the model yourself and want the model without the pretrained weights, you can set `pretrained=False`.

We will publish more trained models soon. Please look forward to it!

### Use Dataset

The `mag-net` package supports **PyTorch** natively by providing a PyTorch dataset. You can get the dataset the following way:

```python
import magnet

dataset = magnet.PyTorchDataset(download_path="data/", download=True)
```

With `download=True`, the data will automatically be downloaded if it does not exist yet locally. 

We also support **TensorFlow** natively by providing a `tf.data` style dataset. You can get the dataset the following way:

```python
import magnet

dataset = magnet.TensorFlowDataset(download_path="data/", download=True)
```

With `download=True`, the data will automatically be downloaded if it does not exist yet locally. 

For other use cases, you must download the dataset manually. The following code will download the dataset to `data/` directory.

```python
import magnet

magnet.download_dataset(download_path="data/")
```


## How to Cite

If you used MagNet, please cite us with the following BibTeX item.

<!-- TODO: Update once dataset paper is published. -->

```
@INPROCEEDINGS{9265869,
  author={H. {Li} and S. R. {Lee} and M. {Luo} and C. R. {Sullivan} and Y. {Chen} and M. {Chen}},
  booktitle={2020 IEEE 21st Workshop on Control and Modeling for Power Electronics (COMPEL)}, 
  title={MagNet: A Machine Learning Framework for Magnetic Core Loss Modeling}, 
  year={2020},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/COMPEL49091.2020.9265869}
}
```

## Sponsors

This work is sponsored by the ARPA-E DIFFERENTIATE Program.

<img src="images/arpae.jpg" width=300>
