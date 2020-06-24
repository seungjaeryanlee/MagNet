#!/usr/bin/env python3
import random

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pywt
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

import analysis


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=1, batch_first=True, bidirectional=False)
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :] # Get last output only (many-to-one)
        x = self.fc_layers(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dataset(data_length=1000, sample_rate=2e-6):
    # Load CSV Files
    # TODO: Use numpy instead of pandas
    I_sin_5k = pd.read_csv('data/raw/I(sin_5k_hiB).csv', header=None).iloc[:,1:data_length+1]
    V_sin_5k = pd.read_csv('data/raw/V(sin_5k_hiB).csv', header=None).iloc[:,1:data_length+1]

    I_tri_5k = pd.read_csv('data/raw/I(tri_5k_hiB).csv', header=None).iloc[:,1:data_length+1]
    V_tri_5k = pd.read_csv('data/raw/V(tri_5k_hiB).csv', header=None).iloc[:,1:data_length+1]

    I_trap_5k = pd.read_csv('data/raw/I(trap_5k_hiB).csv', header=None).iloc[:,1:data_length+1]
    V_trap_5k = pd.read_csv('data/raw/V(trap_5k_hiB).csv', header=None).iloc[:,1:data_length+1]

    I = pd.concat([I_sin_5k , I_tri_5k , I_trap_5k], ignore_index=True)
    V = pd.concat([V_sin_5k , V_tri_5k , V_trap_5k], ignore_index=True)

    # Compute labels
    P = V * I
    t = np.arange(0, (data_length-0.5) * sample_rate, sample_rate)
    Loss_meas = np.trapz(P, t, axis=1) / (sample_rate * data_length)

    # Reshape data
    in_tensors = torch.from_numpy(V.to_numpy()).view(-1, data_length, 1)
    out_tensors = torch.from_numpy(Loss_meas).view(-1, 1)

    # Save data as CSV
    np.save("dataset.lstm.in.npy", in_tensors.numpy())
    np.save("dataset.lstm.out.npy", out_tensors.numpy())

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)


def load_dataset(in_filename="dataset.lstm.in.npy", out_filename="dataset.lstm.out.npy"):
    in_tensors = torch.from_numpy(np.load(in_filename))
    out_tensors = torch.from_numpy(np.load(out_filename))

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)


def main(CONFIG):
    # Reproducibility
    random.seed(CONFIG.SEED)
    np.random.seed(CONFIG.SEED)
    torch.manual_seed(CONFIG.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup CPU or GPU
    if CONFIG.USE_GPU and not torch.cuda.is_available():
        raise ValueError("GPU not detected but CONFIG.USE_GPU is set to True.")
    device = torch.device("cuda" if CONFIG.USE_GPU else "cpu")

    # Setup dataset and dataloader
    # NOTE(seungjaeryanlee): Load saved dataset for speed
    # dataset = get_dataset()
    dataset = load_dataset()
    train_size = int(0.6 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
    kwargs = {'num_workers': 1, 'pin_memory': True} if CONFIG.USE_GPU else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, **kwargs)

    # Setup neural network and optimizer
    net = Net().double().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=CONFIG.LR)
    # Log number of parameters
    CONFIG.NUM_PARAMETERS = count_parameters(net)

    # Setup wandb
    wandb.init(project="MagNet", config=OmegaConf.to_container(CONFIG), reinit=True)
    wandb.watch(net)

    # Training
    for epoch_i in range(1, CONFIG.NUM_EPOCH+1):
        # Train for one epoch
        epoch_train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # Compute Validation Loss
        with torch.no_grad():
            epoch_valid_loss = 0
            for inputs, labels in valid_loader:
                outputs = net(inputs.to(device))
                loss = criterion(outputs, labels.to(device))

                epoch_valid_loss += loss.item()

        print(f"Epoch {epoch_i:2d} "
            f"Train {epoch_train_loss / len(train_dataset):.5f} "
            f"Valid {epoch_valid_loss / len(valid_dataset):.5f}")
        wandb.log({
            "train/loss": epoch_train_loss / len(train_dataset),
            "valid/loss": epoch_valid_loss / len(valid_dataset),
        })

    # Evaluation
    net.eval()
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            y_pred.append(net(inputs.to(device)))
            y_meas.append(labels.to(device))

    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    test_mse_loss = F.mse_loss(y_meas, y_pred).item() / len(test_dataset)
    print(f"Test Loss: {test_mse_loss:.8f}")
    wandb.log({"test/loss": test_mse_loss})

    # Analysis
    wandb.log({
        "test/prediction_vs_target": wandb.Image(analysis.get_scatter_plot(y_pred, y_meas)),
        "test/prediction_vs_target_histogram": wandb.Image(analysis.get_two_histograms(y_pred, y_meas)),
        "test/error_histogram": wandb.Image(analysis.get_error_histogram(y_pred, y_meas)),
    })

    # Close wandb
    wandb.join()

    # Used in `tune_hyperparameters.py`
    return test_mse_loss


if __name__ == "__main__":
    # Load Configuration
    YAML_CONFIG = OmegaConf.load("lstm.yaml")
    CLI_CONFIG = OmegaConf.from_cli()
    CONFIG = OmegaConf.merge(YAML_CONFIG, CLI_CONFIG)

    main(CONFIG)
