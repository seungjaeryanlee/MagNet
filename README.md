# MagNet

![wandb](https://img.shields.io/badge/wandb-metric-yellow)

MagNet is a large-scale dataset designed to enable researchers modeling magnetic core loss using machine learning to accelerate the design process of power electronics. The dataset contains a large amount of voltage and current data of different magnetic components with different shapes of waveforms and different properties measured in the real world. Researchers may use these data as pairs of excitations and responses to build up dynamic magnetic models or calculate the core loss to derive static models.



```
📂 docs                 # Code for the website
📂 data                 # Folder for data preprocessing
```

## Algorithms

- [x] Fully Connected Layer: `run_fc.py`, `fc.yaml`
- [x] 1D Convolutional Layer: `run_conv1d.py`, `conv1d.yaml`
- [x] Wavelet Transform + Conv2D: `run_wavelet.py`, `wavelet.yaml`
- [x] LSTM: `lstm.py`, `lstm.yaml`

