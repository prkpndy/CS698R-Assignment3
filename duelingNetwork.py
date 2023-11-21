import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import device


def createDuelingNetwork(inDim, outDim, hDim={'valueStream': [32, 32], 'advantageStream': [32, 32]}, activation=F.relu):
    # This creates a Feed Forward Neural Network class and instantiates it and returns the class
    # The class should be derived from torch nn.Module and it should have init and forward method at the very least
    # The forward function should return q-value which is derived internally from action-advantage function and v-function,
    # Note we center the advantage values, basically we subtract the mean from each state-action value

    class D3QN(nn.Module):
        def __init__(self, inDim, outDim, hDim, activation):
            super(D3QN, self).__init__()

            self.inDim = inDim
            self.outDim = outDim
            self.hDimValueStream = hDim['valueStream']
            self.hDimAdvantageStream = hDim['advantageStream']

            self.numHiddenLayersValueStream = len(self.hDimValueStream)
            self.numHiddenLayersAdvantageStream = len(self.hDimAdvantageStream)
            self.activation = activation
            self.hiddenLayersValueStream = nn.ModuleList()
            self.hiddenLayersAdvantageStream = nn.ModuleList()

            self.hiddenLayersValueStream.append(
                nn.Linear(self.inDim, self.hDimValueStream[0]))
            self.hiddenLayersAdvantageStream.append(
                nn.Linear(self.inDim, self.hDimAdvantageStream[0]))

            for i in range(1, self.numHiddenLayersValueStream):
                self.hiddenLayersValueStream.append(
                    nn.Linear(self.hDimValueStream[i-1], self.hDimValueStream[i]))

            for i in range(1, self.numHiddenLayersAdvantageStream):
                self.hiddenLayersAdvantageStream.append(
                    nn.Linear(self.hDimAdvantageStream[i-1], self.hDimAdvantageStream[i]))

            self.outputLayerValueStream = nn.Linear(
                self.hDimValueStream[-1], 1)
            self.outputLayerAdvantageStream = nn.Linear(
                self.hDimAdvantageStream[-1], self.outDim)

        def forward(self, x):
            x = x.to(device)

            value = self.activation(self.hiddenLayersValueStream[0](x))
            advantage = self.activation(self.hiddenLayersAdvantageStream[0](x))
            for i in range(1, self.numHiddenLayersValueStream):
                value = self.activation(self.hiddenLayersValueStream[i](value))
            for i in range(1, self.numHiddenLayersAdvantageStream):
                advantage = self.activation(
                    self.hiddenLayersAdvantageStream[i](advantage))

            value = self.outputLayerValueStream(value)
            advantage = self.outputLayerAdvantageStream(advantage)

            qValues = value + (advantage - advantage.mean())

            return qValues

    duelNetwork = D3QN(inDim, outDim, hDim, activation)

    return duelNetwork
