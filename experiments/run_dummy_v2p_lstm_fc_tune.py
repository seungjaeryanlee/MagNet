#!/usr/bin/env python3
import random

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import wandb
import optuna


class Net(nn.Module):
    def __init__(self,trial):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(1, 42, num_layers=1, batch_first=True, bidirectional=False)
        
        n_units_FC4 = trial.suggest_int("n_units_FC4", 4, 64)
        self.fc_layers = nn.Sequential(
            nn.Linear(42, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.ReLU(),
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, n_units_FC4),
            nn.ReLU(),
            nn.Linear(n_units_FC4, 1)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :] # Get last output only (many-to-one)
        x = self.fc_layers(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dataset(data_length=400, sample_rate=2e-6):
    # Load CSV Files
    # TODO: Use numpy instead of pandas
    
    V = pd.read_csv('D:\Dropbox (Princeton)\MagNet_TestData_dummy\Dummy_volt.csv', header=None).iloc[:,0:data_length]
    # B = pd.read_csv('D:\Dropbox (Princeton)\MagNet_TestData_dummy\Dummy_flux.csv', header=None)
    # F = pd.read_csv('D:\Dropbox (Princeton)\MagNet_TestData_dummy\Dummy_freq.csv', header=None)
    P = pd.read_csv('D:\Dropbox (Princeton)\MagNet_TestData_dummy\Dummy_loss.csv', header=None)

    # F = F.apply(np.log10)
    # B = B.apply(np.log10)
    P = P.apply(np.log10)

    I = pd.concat([P], axis=1, ignore_index=True)

    # Reshape data
    in_tensors = torch.from_numpy(V.to_numpy()).view(-1, data_length, 1)
    out_tensors = torch.from_numpy(I.to_numpy()).view(-1, 1)

    # Save data as CSV
    np.save("dataset.lstm.in.npy", in_tensors.numpy())
    np.save("dataset.lstm.out.npy", out_tensors.numpy())

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)


def load_dataset(in_filename="dataset.lstm.in.npy", out_filename="dataset.lstm.out.npy"):
    in_tensors = torch.from_numpy(np.load(in_filename))
    out_tensors = torch.from_numpy(np.load(out_filename))

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)


def objective(trial):
    # Load Configuration
    YAML_CONFIG = OmegaConf.load("lstm.yaml")
    CLI_CONFIG = OmegaConf.from_cli()
    CONFIG = OmegaConf.merge(YAML_CONFIG, CLI_CONFIG)

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
    dataset = get_dataset()
    #dataset = load_dataset()
    train_size = int(0.6 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
    kwargs = {'num_workers': 1, 'pin_memory': True} if CONFIG.USE_GPU else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, **kwargs)

    # Setup neural network and optimizer
    net = Net(trial).double().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # Log number of parameters
    CONFIG.NUM_PARAMETERS = count_parameters(net)

    # Setup wandb
    # wandb.init(project="MagNet", config=CONFIG)
    # wandb.watch(net)

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
        # wandb.log({
        #     "train/loss": epoch_train_loss / len(train_dataset),
        #     "valid/loss": epoch_valid_loss / len(valid_dataset),
        # })

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
    # print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item() / len(test_dataset):.8f}")
    # wandb.log({"test/loss": F.mse_loss(y_meas, y_pred).item() / len(test_dataset)})

    # Predicton vs Target Plot
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    ax.scatter(y_meas.cpu().numpy(), y_pred.cpu().numpy(), label="Prediction")
    ax.plot(y_meas.cpu().numpy(), y_meas.cpu().numpy(), 'k--', label="Target")
    ax.grid(True)
    ax.legend()
    # wandb.log({"prediction_vs_target": wandb.Image(fig)})
    
    # Relative Error
    Error_re1 = abs(y_pred.cpu().numpy()-y_meas.cpu().numpy())/abs(y_meas.cpu().numpy())*100
    Error_re1[np.where(Error_re1>500)] = 500
    # Error_re_max1 = np.max(Error_re1);
    Error_re_avg1 = np.mean(Error_re1);
    # print(f"Relative Error: {Error_re_avg1:.8f}")
    # wandb.log({"Relative Error": Error_re_avg1})
    
    # np.savetxt("pred.csv", y_pred.cpu().numpy())
    # np.savetxt("meas.csv", y_meas.cpu().numpy())
    return Error_re_avg1


if __name__ == "__main__":
    study = optuna.create_study(
        study_name = 'LSTM-FC',
        direction="minimize",
        storage = 'sqlite:///example.db',
        load_if_exists=True)
    study.optimize(objective, n_trials=3)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
