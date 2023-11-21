import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import device


def createValueNetwork(inDim, outDim, hDim=[32, 32], activation=F.relu):
    # This creates a Feed Forward Neural Network class and instantiates it and returns the class
    # The class should be derived from torch nn.Module and it should have init and forward method at the very least
    # The forward function should return q-value for each possible action

    class DQN(nn.Module):
        def __init__(self, inDim, outDim, hDim, activation):
            super(DQN, self).__init__()

            self.inDim = inDim
            self.outDim = outDim
            self.hDim = hDim

            self.numHiddenLayers = len(hDim)
            self.activation = activation
            self.hiddenLayers = nn.ModuleList()

            self.hiddenLayers.append(nn.Linear(inDim, hDim[0]))

            for i in range(1, self.numHiddenLayers):
                self.hiddenLayers.append(nn.Linear(hDim[i-1], hDim[i]))

            self.outputLayer = nn.Linear(hDim[-1], outDim)

        def forward(self, x):
            x = x.to(device)

            val = self.activation(self.hiddenLayers[0](x))
            for i in range(1, self.numHiddenLayers):
                val = self.activation(self.hiddenLayers[i](val))

            return self.outputLayer(val)

    valueNetwork = DQN(inDim, outDim, hDim, activation)

    return valueNetwork
