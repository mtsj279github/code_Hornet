import torch.nn as nn


class res_block(nn.Module):
    def __init__(self, in_channels, out_channels, channels_change=False,
                 stride=1):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1,
                               stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1,
                               stride=1)
        if channels_change:
            self.conv3 = nn.Conv2d(in_channels, out_channels, 1,
                                   stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.conv3:
            x = self.conv3(x)
        return self.relu(x + y)


def res_layer(in_channels, out_channels, num_residuals, first_layer=False):
    if first_layer:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_layer:
            blk.append(res_block(in_channels, out_channels,
                                 channels_change=True, stride=2))
        else:
            blk.append(res_block(out_channels, out_channels))
    return nn.Sequential(*blk)
