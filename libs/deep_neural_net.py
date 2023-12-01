import torch.nn
from collections import OrderedDict
from torch.optim.lr_scheduler import OneCycleLR


class DeepNeuralNet(torch.nn.Module):
    # NL: the number of hidden layers
    # NN: the number of vertices in each layer
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 depth,
                 act=torch.nn.Tanh):
        super(DeepNeuralNet, self).__init__()
        layers = [('input', torch.nn.Linear(input_size, hidden_size)), ('input_activation', act())]
        for i in range(depth):
            layers.append(
                ('hidden_%d' % i, torch.nn.Linear(hidden_size, hidden_size))
            )
            layers.append(('activation_%d' % i, act()))
        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))

        layer_dict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layer_dict)

    def forward(self, x):
        out = self.layers(x)
        return out
