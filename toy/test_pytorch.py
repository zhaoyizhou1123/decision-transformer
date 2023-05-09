import torch
import torch.nn as nn
import numpy as np

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2,3)
        # self.layer.weight.data.fill_(0)
        # self.layer.bias.data.fill_(0)
        
    def show_weights(self):
        print(f"Layer weights: {self.layer.weight}")
        print(f"Layer bias: {self.layer.bias}")

    def forward(self, x):
        return self.layer(x)

model = Network()
for param in model.parameters():
    param.requires_grad = False
    param.data.fill_(0)
    print(f"param = {param}, type {type(param)}, size {param.size()}")
model.show_weights()

