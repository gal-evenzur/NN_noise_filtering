import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Structure for the NN: #
class Noise_reductor(nn.Module):
    def __init__    (self, f_signal=5001, h1=1000, clean_sig=5001):
        super().__init__()
        self.fc1 = nn.Linear(f_signal, h1)
        self.out = nn.Linear(h1, clean_sig)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.out(x))

        return x

# Now we want to train the parameters #
# First, IMPORTING DATA #

with open("data.json", "r") as f:
    data = json.load(f)

X_train = data["train"]["f_signals"]
Y_train = data["train"]["f_cSignal"]

X_test = data["test"]["f_signals"]
Y_test = data["test"]["f_cSignal"]

X_train = torch.FloatTensor(X_train)
Y_train = torch.FloatTensor(Y_train)

X_test = torch.FloatTensor(X_test)
Y_test = torch.FloatTensor(Y_test)

t = torch.FloatTensor(data["t"])
f = torch.FloatTensor(data["f"])

# TEST THROUGH NN #
model = Noise_reductor()
Y_pred = model(X_train)

'''
#    PLOTTING THE FILTERING TO SEE IF IT WORKED   #
fig, sigPlot = plt.subplots(nrows=2, ncols=5)

for i in range(5):
    sigPlot[0][i].semilogy(f, X_train[i].detach(), ".")
    sigPlot[1][i].semilogy(f, Y_train[i].detach(), "r.")


plt.show()
'''