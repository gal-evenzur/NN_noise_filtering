# %% IMPORTS #

import torch
import json
import torch.nn as nn
import torch.nn.functional as F
# optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.
from torch.optim import SGD

import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for testing headless

import matplotlib.pyplot as plt
import numpy as np

def scale_tensor(dat_raw, device):
    dat_raw = torch.FloatTensor(dat_raw)
    min = dat_raw.amin(1)
    max = dat_raw.amax(1)
    # print(f"data is {dat_raw.shape} - {min.shape} / ({max.shape} - {min.shape} ")

    dat_norm = (dat_raw - min[:, None])/(max[:, None] - min[:, None])
    std = dat_norm.std(1)
    dat_norm = dat_norm/std[:, None]
    return dat_norm.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% Structure for the NN: #
class Noise_reductor(nn.Module):
    def __init__(self, f_signal=5001, h1=1500, h2=1000, h3=750, h4=500, h5=250, h6=100):
        super().__init__()

        self.linear_stack = nn.Sequential(
          nn.Linear(f_signal, h1),
          nn.Linear(h1, h2),
          nn.Linear(h2, h3),
          nn.Linear(h3, h6),
          nn.Linear(h6, f_signal), # output
        )


    def forward(self, x):
      logits = self.linear_stack(x)
      return logits



# Now we want to train the parameters #
# %% First, IMPORTING DATA #

with open("model_creating_data/data.json", "r") as f:
    data = json.load(f)

X_train_raw = data["train"]["f_signals"]
Y_train_raw = data["train"]["f_cSignal"]

X_test_raw = data["test"]["f_signals"]
Y_test_raw = data["test"]["f_cSignal"]

# Turn into a torch tensor and then normalize it std=1, mean=0
X_train = scale_tensor(X_train_raw, device)
Y_train = scale_tensor(Y_train_raw, device)

X_test = scale_tensor(X_test_raw, device)
Y_test = scale_tensor(Y_test_raw,device)



t = torch.FloatTensor(data["t"])
f = torch.FloatTensor(data["f"])


'''
# Example of the scaled versions of the signals: #
fig, sigPlot = plt.subplots(nrows=2, ncols=5)


for i in range(5):
    X_t_numpy = X_train[i].cpu().detach().numpy()
    Y_t_numpy = Y_train[i].cpu().detach().numpy()

    X_train_plot = X_t_numpy
    Y_train_plot = Y_t_numpy


    sigPlot[0][i].semilogy(f, X_train_plot, ".")
    sigPlot[1][i].semilogy(f, Y_train_plot, "r.")

    sigPlot[0][i].set_xlim(1950,2050)
    sigPlot[0][i].set_ylim(0.00001,10)
    sigPlot[1][i].set_xlim(1950,2050)
    sigPlot[1][i].set_ylim(0.00001,10)

plt.show()
'''

# %% TEST THROUGH NN #
model = Noise_reductor().to(device)

optimizer = SGD(model.parameters(), lr = 0.01)
criterion = nn.MSELoss()

epochs = 3500
losses = []
eps = []
model.train()
for ep in range(epochs):
    Y_pred = model(X_train)

    # Measure the loss/error, gonna be high at first
    loss = criterion(Y_pred, Y_train) # predicted values vs the y_train

    # Keep Track of our losses

    # print every 10 epoch
    if ep % (epochs/10) == 0:
        losses.append(loss.cpu().detach().numpy())
        eps.append(ep)
        # print(f'Epoch: {ep} and loss: {loss}')

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

# %%   PLOTTING THE FILTERING TO SEE IF IT WORKED   #

plt.figure()
plt.semilogy(eps, losses)


n = 10
fig, sigPlot = plt.subplots(nrows=2, ncols=n)

Y_pred = model(X_test)

start = 10
for i_ in range(start,start+n):

    Y_p_numpy = Y_pred[i_].cpu().detach().numpy()
    Y_t_numpy = Y_test[i_].cpu().detach().numpy()
    X_t_numpy = X_test[i_].cpu().detach().numpy()

    Y_pred_plot = Y_p_numpy
    Y_train_plot = Y_t_numpy

    i = i_%n


    sigPlot[0][i].semilogy(f, Y_pred_plot, ".")
    sigPlot[0][i].semilogy(f, Y_train_plot, "r.")
    sigPlot[1][i].semilogy(f, X_t_numpy, "*")


    xbor = [1000, 3000]
    sigPlot[0][i].set_xlim(xbor[0],xbor[1])
    sigPlot[0][i].set_ylim(0.000001,10)
    sigPlot[0][i].set_xticks(np.linspace(xbor[0], xbor[1], 3))  # 11 ticks between 1950 and 2050

plt.tight_layout()
plt.show()