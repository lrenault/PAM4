import torch
import torch.nn as nn

import utils

filename = 'audios/mix.wav'
nb_seconds = 3
nb_epoch = 200

audio, sample_rate = utils.load(filename)
audio = audio[:, :sample_rate * nb_seconds]

nb_mixtures, T = audio.size()
nb_sources = nb_mixtures

B = nn.Linear(nb_mixtures, nb_sources, bias=False)
estim_function = utils.H(utils.edgeworth, nb_sources)

#%%
def train(model, mixtures, optimizer, criterion, epoch):
    model.train()
    y = torch.t(model(torch.t(mixtures)))
    # compute loss
    loss = criterion(y)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

#%% optimization definition
model = B
criterion = utils.estimation_equation(nb_sources, estim_function)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)

for epoch in range(nb_epoch):
    loss = train(model, audio, optimizer, criterion, epoch)
    print(loss)