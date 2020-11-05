import torch.nn as nn


class MiniLSTM(nn.Module):
    def __init__(self, pretrained=True):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=1, batch_first=True, bidirectional=False)
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        if pretrained:
            self.load_state_dict(torch.load("MiniLSTM.pt"))

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :] # Get last output only (many-to-one)
        x = self.fc_layers(x)
        return x
