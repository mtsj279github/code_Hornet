import torch.nn as nn
import Res_block
import simple_function

net = nn.Sequential(
    nn.Conv2d(3, 64, 7, 2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(3, 2, 1))

net.add_module('res_block_1', Res_block.res_layer(64, 64, 2, first_layer=True))
net.add_module('res_block_2', Res_block.res_layer(64, 128, 2))
net.add_module('res_block_3', Res_block.res_layer(128, 256, 2))
net.add_module('res_block_4', Res_block.res_layer(256, 512, 2))
net.add_module('GAP', simple_function.GAP2D())
net.add_module('fc', nn.Sequential(simple_function.Flattenlayer(),
                                   nn.Linear(512, 2)))



