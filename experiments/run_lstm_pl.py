#!/usr/bin/env python3
import random

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytorch_lightning as pl
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

import analysis


class Net(pl.LightningModule):
    def __init__(self, CONFIG):
        super(Net, self).__init__()
        # TODO: Use self.save_hyperparameters()
        self.CONFIG = CONFIG
        self.criterion = nn.MSELoss()
        self.lstm = nn.LSTM(1, 64, num_layers=1, batch_first=True, bidirectional=False)
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def prepare_data(self):
        # Load dataset
        in_tensors = torch.from_numpy(np.load("dataset.lstm.in.npy"))
        out_tensors = torch.from_numpy(np.load("dataset.lstm.out.npy"))
        dataset = torch.utils.data.TensorDataset(in_tensors, out_tensors)

        # Split dataset
        train_size = int(0.6 * len(dataset))
        valid_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - valid_size
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :] # Get last output only (many-to-one)
        x = self.fc_layers(x)
        return x

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.CONFIG.BATCH_SIZE,
            shuffle=True,
            num_workers=6,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.CONFIG.BATCH_SIZE,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.CONFIG.BATCH_SIZE,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.CONFIG.LR)

    def training_step(self, batch, batch_idx):
        """
        Train the model with one mini-batch.
        """
        inputs, labels = batch
        outputs = self(inputs)
        return { "loss": self.criterion(outputs, labels), "log": {} }

    def training_epoch_end(self, outputs):
        epoch_train_loss = sum([output["loss"].item() for output in outputs]) / len(self.train_dataset)
        print(f"Epoch {self.current_epoch} Loss {epoch_train_loss:.4f}")
        wandb.log({ "train/loss": epoch_train_loss })
        return { "train/loss": epoch_train_loss, 'log': {} }

    def validation_step(self, batch, batch_nb):
        with torch.no_grad():
            inputs, labels = batch
            outputs = self(inputs)
            return { "loss": self.criterion(outputs, labels), "log": {} }

    def validation_epoch_end(self, outputs):
        epoch_valid_loss = sum([output["loss"].item() for output in outputs]) / len(self.valid_dataset)
        print(f"Epoch {self.current_epoch} Loss {epoch_valid_loss:.4f}")
        wandb.log({ "valid/loss": epoch_valid_loss })
        return { "valid/loss": epoch_valid_loss, 'log': {} }

    def test_step(self, batch, batch_nb):
        with torch.no_grad():
            return { "y_pred": self(inputs), "y_meas": labels}

    def test_epoch_end(self, outputs):
        y_pred = [output["y_pred"] for output in outputs]
        y_meas = [output["y_meas"] for output in outputs]
        y_pred = torch.cat(y_pred, dim=0)
        y_meas = torch.cat(y_meas, dim=0)

        test_mse_loss = F.mse_loss(y_meas, y_pred).item() / len(test_dataset)
        print(f"Test Loss: {test_mse_loss:.8f}")
        wandb.log({"test/loss": test_mse_loss})

        # Analysis
        wandb.log({
            "test/prediction_vs_target": wandb.Image(analysis.get_scatter_plot(y_pred, y_meas)),
            "test/prediction_vs_target_histogram": wandb.Image(analysis.get_two_histograms(y_pred, y_meas)),
            "test/error_histogram": wandb.Image(analysis.get_error_histogram(y_pred, y_meas)),
        })

        return { "test/loss": test_mse_loss, 'log': {} }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(CONFIG):
    # Reproducibility
    random.seed(CONFIG.SEED)
    np.random.seed(CONFIG.SEED)
    torch.manual_seed(CONFIG.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup model
    net = Net(CONFIG).double()

    # Setup wandb
    wandb.init(project="MagNet", config=OmegaConf.to_container(CONFIG), reinit=True)
    wandb.watch(net)

    # Log number of parameters
    CONFIG.NUM_PARAMETERS = count_parameters(net)
    trainer = pl.Trainer(
        # TODO: Add CONFIG parameters for devices
        gpus=1,
        # Don't show progress bar
        progress_bar_refresh_rate=0,
        check_val_every_n_epoch=1,
        # TODO: Try early stopping
        max_epochs=CONFIG.NUM_EPOCH,
    )
    trainer.fit(net)

    # Close wandb
    wandb.join()

    # Used in `tune_hyperparameters.py`
    # TODO: How to get numbers?
    # return test_mse_loss


if __name__ == "__main__":
    # Load Configuration
    YAML_CONFIG = OmegaConf.load("lstm.yaml")
    CLI_CONFIG = OmegaConf.from_cli()
    CONFIG = OmegaConf.merge(YAML_CONFIG, CLI_CONFIG)

    main(CONFIG)
