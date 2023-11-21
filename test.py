import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class Net(nn.Module):
    def __init__(self, inDim, outDim, hDim, activation):
        super(Net, self).__init__()

        self.inDim = inDim
        self.outDim = outDim
        self.hDim = hDim

        self.numHiddenLayers = len(hDim)
        self.activation = activation
        self.hiddenLayers = nn.ModuleList()

        self.hiddenLayers.append(nn.Linear(inDim, hDim[0]))

        # self.hiddenLayers = nn.Linear(inDim, hDim[0])
        for i in range(1, self.numHiddenLayers):
            self.hiddenLayers.append(nn.Linear(hDim[i-1], hDim[i]))

        self.outputLayer = nn.Linear(hDim[-1], outDim)

    def forward(self, x):
        print(x.device)
        val = self.activation(self.hiddenLayers[0](x))
        for i in range(1, self.numHiddenLayers):
            val = self.activation(self.hiddenLayers[i](val))

        return self.outputLayer(val)


net = Net(4, 4, [32], F.relu).to(device)
print(net.named_parameters())

for name, value in net.named_parameters():
    print(name)
    print(value)

# input = torch.randn((10, 4), device=device)
# print('input: ', input)

# with torch.no_grad():
#     out = net(input)
# print('output: ', out)

# print(out.max(1)[1].view)
# print(out.max(0)[1].view(1, 1))
