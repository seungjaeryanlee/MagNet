import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#%%========== Import V-I Data ================================================
data_length = 1000
sample_rate = 2e-6

t = np.arange(0, (data_length-0.5) * sample_rate, sample_rate)

# I_sin_1k = pd.read_csv('data/raw/I(sin_1k_hiB).csv',header=None).iloc[:,1:data_length+1]
# I_sin_2k = pd.read_csv('data/raw/I(sin_2k_hiB).csv',header=None).iloc[:,1:data_length+1]
I_sin_5k = pd.read_csv('data/raw/I(sin_5k_hiB).csv',header=None).iloc[:,1:data_length+1]
# V_sin_1k = pd.read_csv('data/raw/V(sin_1k_hiB).csv',header=None).iloc[:,1:data_length+1]
# V_sin_2k = pd.read_csv('data/raw/V(sin_2k_hiB).csv',header=None).iloc[:,1:data_length+1]
V_sin_5k = pd.read_csv('data/raw/V(sin_5k_hiB).csv',header=None).iloc[:,1:data_length+1]

# I_tri_1k = pd.read_csv('data/raw/I(tri_1k_hiB).csv',header=None).iloc[:,1:data_length+1]
# I_tri_2k = pd.read_csv('data/raw/I(tri_2k_hiB).csv',header=None).iloc[:,1:data_length+1]
I_tri_5k = pd.read_csv('data/raw/I(tri_5k_hiB).csv',header=None).iloc[:,1:data_length+1]
# V_tri_1k = pd.read_csv('data/raw/V(tri_1k_hiB).csv',header=None).iloc[:,1:data_length+1]
# V_tri_2k = pd.read_csv('data/raw/V(tri_2k_hiB).csv',header=None).iloc[:,1:data_length+1]
V_tri_5k = pd.read_csv('data/raw/V(tri_5k_hiB).csv',header=None).iloc[:,1:data_length+1]

# I_trap_1k = pd.read_csv('data/raw/I(trap_1k_hiB).csv',header=None).iloc[:,1:data_length+1]
# I_trap_2k = pd.read_csv('data/raw/I(trap_2k_hiB).csv',header=None).iloc[:,1:data_length+1]
I_trap_5k = pd.read_csv('data/raw/I(trap_5k_hiB).csv',header=None).iloc[:,1:data_length+1]
# V_trap_1k = pd.read_csv('data/raw/V(trap_1k_hiB).csv',header=None).iloc[:,1:data_length+1]
# V_trap_2k = pd.read_csv('data/raw/V(trap_2k_hiB).csv',header=None).iloc[:,1:data_length+1]
V_trap_5k = pd.read_csv('data/raw/V(trap_5k_hiB).csv',header=None).iloc[:,1:data_length+1]

I = pd.concat([I_sin_5k , I_tri_5k , I_trap_5k], ignore_index=True)
V = pd.concat([V_sin_5k , V_tri_5k , V_trap_5k], ignore_index=True)

#%%========== Display Scalogram (uncomment when needed) ======================
signal = V_trap_5k.iloc[100, 1:1002]

wave_name = 'cgau8'
total_scale = 30
fc = pywt.central_frequency(wave_name)
fmax = 10e3
cparam = (1 / sample_rate) / fmax * fc * total_scale
scales = cparam / np.arange(total_scale, 1, -1)
[cwtmatr, frequencies] = pywt.cwt(signal, scales, wave_name, sample_rate)

# plt.figure(figsize=(8, 4))
# plt.subplot(211)
# plt.contourf(t[1:1000], frequencies, abs(cwtmatr))
# plt.subplot(212)
# image_size = 24
# plt.contourf(resize(abs(cwtmatr), (image_size, image_size)))
# plt.show()

#%%========== Compute Measured Loss ==========================================
P = V * I
data_size = np.size(P, 0)
Loss_meas = np.zeros(data_size)
for index, row in P.iterrows():
    Loss_meas[index] = np.trapz(row, t) / (sample_rate * data_length)

#%%========== Create Scalograms ==============================================
wave_name = 'cgau8'
total_scale = 30
fc = pywt.central_frequency(wave_name)
fmax = 10e3
cparam = (1 / sample_rate) / fmax * fc * total_scale
scales = cparam / np.arange(total_scale, 1, -1)

image_size = 24
scalogram = np.zeros([data_size, image_size, image_size])
for index, row in V.iterrows():
    [cwtmatr, frequencies] = pywt.cwt(row, scales, wave_name, sample_rate)
    scalogram[index:, :, ] = resize(abs(cwtmatr), (image_size, image_size))
    print(f"{index} finished")

#%%========== Config the Neural Network ======================================
# function to count number of parameters
# def get_n_params(model):
#     np=0
#     for p in list(model.parameters()):
#         np += p.nelement()
#     return np

input_size  = image_size * image_size   # images in NxN pixels
output_size = 1      # 1 regression output

Dataset = torch.utils.data.TensorDataset(torch.from_numpy(scalogram), torch.from_numpy(Loss_meas))
train_size = int(0.7 * data_size)
test_size = data_size - train_size
train_db, test_db = torch.utils.data.random_split(Dataset, [train_size, test_size])
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_db, batch_size=1, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=2)
        self.batch_norm3 = nn.BatchNorm2d(16)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(256, 8)
        self.fc2 = nn.Linear(8, 1)
    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        # x = self.drop_out(x)
        x = self.fc2(x)
        return x

net = Net().double()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

#%%========== Train the Neural Network =======================================
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.view(-1, 1, 24, 24)
        labels = labels.view(-1, 1)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

print('Finished Training')

# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

# net = Net()
# net.load_state_dict(torch.load(PATH))

#%%========== Test the Neural Network ========================================
net.eval()
with torch.no_grad():
    y_meas = np.zeros(test_size)
    y_pred = np.zeros(test_size)
    index = 0
    for index, (inputs, labels) in enumerate(test_loader, 0):
        inputs = inputs.view(-1, 1, 24, 24)
        y_pred[index] = net(inputs)
        y_meas[index] = labels
        
# plt.figure(figsize = (8,8))
# plt.subplot()
# plt.scatter(y_meas, y_pred)
# plt.plot(y_meas, y_meas, 'k--')
# plt.grid(True)
# plt.show()
