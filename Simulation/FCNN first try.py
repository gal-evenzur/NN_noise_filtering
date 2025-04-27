import torch
import json
import torch.nn as nn
import torch.nn.functional as F
# optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.
from torch.optim import SGD

import matplotlib.pyplot as plt

def log_normlised_tensor(dat_raw):
    dat_raw = torch.FloatTensor(dat_raw)
    dat_logscaled = torch.log1p(dat_raw)
    dat_logscaled = dat_raw
    mean = dat_logscaled.mean()
    std = dat_logscaled.std()

    dat_norm = (dat_logscaled - mean)/std
    return dat_norm

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

X_train_raw = data["train"]["f_signals"]
Y_train_raw = data["train"]["f_cSignal"]

X_test = data["test"]["f_signals"]
Y_test = data["test"]["f_cSignal"]

# Turn into a torch tensor and then normalize it std=1, mean=0
X_train = log_normlised_tensor(X_train_raw)
Y_train = log_normlised_tensor(Y_train_raw)

X_test = torch.FloatTensor(X_test)
Y_test = torch.FloatTensor(Y_test)



t = torch.FloatTensor(data["t"])
f = torch.FloatTensor(data["f"])

# TEST THROUGH NN #
model = Noise_reductor()


optimizer = SGD(model.parameters(), lr = 0.1)
criterion = nn.MSELoss()

epochs = 200
losses = []
eps = []
model.train()
for ep in range(epochs):
    Y_pred = model(X_train)

    # Measure the loss/error, gonna be high at first
    loss = criterion(Y_pred, Y_train) # predicted values vs the y_train

    # Keep Track of our losses

    # print every 10 epoch
    if ep % 50 == 0:
        losses.append(loss.detach().numpy())
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


plt.figure()
plt.semilogy(eps, losses)


#    PLOTTING THE FILTERING TO SEE IF IT WORKED   #
fig, sigPlot = plt.subplots(nrows=2, ncols=5)

Y_pred = model(X_train)

# Y_pred_plot = Y_pred + abs(Y_pred.min())
# Y_train_plot = Y_train + abs(Y_train.min())

for i in range(5):

    Y_pred_plot = Y_pred[i].detach() + abs(Y_pred[i].detach().min())
    Y_train_plot = Y_train[i].detach() + abs(Y_train[i].detach().min())


    sigPlot[0][i].semilogy(f, Y_pred_plot, ".")
    sigPlot[1][i].semilogy(f, Y_train_plot, "r.")


plt.show()
