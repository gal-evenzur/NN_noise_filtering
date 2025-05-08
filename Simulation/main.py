# %% IMPORTS #

import torch
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# optim contains many optimizers. Here, we're using SGD, stochastic gradient descent..
from torch.optim import SGD, Adam

import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for testing headless

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% Structure for the NN: #

class Noise_reductor(nn.Module):
    def __init__(self, f_signal=5001, h1=1500, h2=1000, h3=750, h4=500, h5=250, h6=100):
        super().__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(f_signal, h1),
            nn.Linear(h1, h3),
            nn.Linear(h3, h6),
            nn.Linear(h6, f_signal), # output
        )


    def forward(self, x):
      logits = self.linear_stack(x)
      return logits


# Data handeling:
def scale_tensor(dat_raw):
    dat_raw = torch.FloatTensor(dat_raw)
    min = dat_raw.amin(1)
    max = dat_raw.amax(1)
    # print(f"data is {dat_raw.shape} - {min.shape} / ({max.shape} - {min.shape} ")

    dat_norm = (dat_raw - min[:, None])/(max[:, None] - min[:, None])
    # std = dat_norm.std(1)
    # dat_norm = dat_norm/std[:, None]
    return dat_norm


class SignalDataset(Dataset):
    def __init__(self, json_path, split="train", transfrom=scale_tensor):

        with open(json_path, "r") as f:
            dat = json.load(f)
        x_raw = dat[split]["f_signals"]
        y_raw = dat[split]["f_cSignal"]

        self.X = transfrom(x_raw)
        self.Y = transfrom(y_raw)

        self.f = torch.FloatTensor(dat["f"])

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_freqs(self):
        return self.f



# Now we want to train the parameters #
# %% First, IMPORTING DATA #

trainSet = SignalDataset('model_creating_data/data.json')
testSet = SignalDataset('model_creating_data/data.json', split="test")

dataloader = DataLoader(trainSet, batch_size=64, shuffle=True)

'''
# Example of the scaled versions of the signals: #
fig, sigPlot = plt.subplots(nrows=2, ncols=5)


x_batch, y_batch = next(iter(dataloader))
f = trainSet.get_freqs()

for i in range(5):

    sigPlot[0][i].semilogy(f, x_batch[i], ".")
    sigPlot[1][i].semilogy(f, y_batch[i], "r.")

    sigPlot[0][i].set_xlim(1950,2050)
    sigPlot[0][i].set_ylim(0.00001,10)
    sigPlot[1][i].set_xlim(1950,2050)
    sigPlot[1][i].set_ylim(0.00001,10)

plt.show()
'''

# %% TRAIN THROUGH NN #

losses = []
eps = []
model = Noise_reductor().to(device)
def train(epochs, optim=SGD, device="cpu", rate = 0.001):
    optimizer = optim(model.parameters(), lr = rate)
    criterion = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        for i, (X_batch, Y_batch) in enumerate(dataloader):
            X = X_batch.to(device)
            Y = Y_batch.to(device)
            Y_pred = model(X)

            # Measure the loss/error, gonna be high at first
            loss = criterion(Y_pred, Y) # predicted values vs the y_train


            # Do some back propagation: take the error rate of forward propagation and feed it back
            # thru the network to fine tune the weights
            optimizer.zero_grad() ## This zeroes out the gradient stored in "model".
                                  ## Remember, by default, gradients are added to the previous step (the gradients are accumulated),
                                  ## and we took advantage of this process to calculate the derivative one data point at a time.
                                  ## NOTE: "optimizer" has access to "model" because of how it was created with the call
                                  ## (made earlier): optimizer = SGD(model.parameters(), lr=0.1).
                                  ## ALSO NOTE: Alternatively, we can zero out the gradient with model.zero_grad().
            loss.backward()
            optimizer.step()
        if loss.detach() < 10**-10:
            break

            # Keep Track of our losses
        if ep % (epochs/300) == 0:
            losses.append(loss.cpu().detach().numpy())
            eps.append(ep)



if __name__ == "__main__":
    train(2000, Adam, device)

# %%   PLOTTING THE FILTERING TO SEE IF IT WORKED   #

plt.figure()
plt.semilogy(eps, losses)
plt.title("Loss vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")

n = 10
fig, sigPlot = plt.subplots(nrows=2, ncols=n,  figsize=(3*n,6))
fig.text(0.6, 0.01, 'f [Hz]', ha='center', fontsize=12)
fig.text(0.01, 0.5, 'FFT', ha='center', rotation='vertical', fontsize=12)
import matplotlib.lines as mlines

# Create proxy artists
data_handle = mlines.Line2D([], [], color='red', marker='.', linestyle='None', label='Data')
pred_handle = mlines.Line2D([], [], color='blue', marker='*', linestyle='None', label='Prediction')

# Add a figure-level legend
fig.legend(handles=[pred_handle, data_handle],
           loc='upper center', ncol=3, fontsize=10, frameon=False)

f = testSet.get_freqs()


start = 10
for i_ in range(start,start+n):
    X_test,Y_test = testSet[i_]
    Y_pred = model(X_test.to(device))

    Y_p_numpy = Y_pred.cpu().detach().numpy()
    Y_t_numpy = Y_test.cpu().detach().numpy()
    X_t_numpy = X_test.cpu().detach().numpy()

    Y_pred_plot = Y_p_numpy
    Y_train_plot = Y_t_numpy

    i = i_%n



    sigPlot[0][i].semilogy(f, Y_pred_plot, "*")
    sigPlot[0][i].semilogy(f, Y_train_plot, "r.")
    sigPlot[1][i].semilogy(f, X_t_numpy, "*")

    # Add a title (number) to each column's top subplot
    sigPlot[0][i].set_title(i)

    xbor = [1970, 2030]
    sigPlot[0][i].set_xlim(xbor[0],xbor[1])
    sigPlot[0][i].set_xticks(np.linspace(xbor[0], xbor[1], 5))  # 11 ticks between 1950 and 2050
    sigPlot[0][i].set_xticklabels([f"{xbor[0]:.0f}", "", "", "", f"{xbor[1]:.0f}"])

plt.tight_layout(rect=[0.03, 0.03, 1, 0.88])
plt.show()