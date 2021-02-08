import numpy as np
import os
import resnet_modle
import simple_function
import torch.nn.init as init
import torch.utils.data as Data

import torchvision.transforms as transforms
import cv2 as cv
import random
import torch
import torch.nn as nn


def image(path):
    feature = []
    for dir_image in os.listdir(path):
        img = cv.imread(path + '\\' + dir_image)
        img = cv.resize(img, (224, 224))
        img = np.swapaxes(img, 2, 0)
        img = img.tolist()
        feature.append(img)
    return feature


count = 168
labels = [0] * count
for i in range(84):
    labels[i] = 1
labels = torch.tensor(labels)
features = image('F:\\Awinter\\MCM\C\\train1')
print(len(features))
features_tensor = torch.Tensor(features)

features_test = image('F:\\Awinter\\MCM\C\\test1')
print(len(features_test))
features_test_tensor = torch.Tensor(features_test)
labels_test = [0] * count
for i in range(84):
    labels_test[i] = 1
labels_test = torch.tensor(labels_test)

batch_size = 16
train_dataset = Data.TensorDataset(features_tensor, labels)
train_iter = Data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

test_dataset = Data.TensorDataset(features_test_tensor, labels_test)
test_iter = Data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

net = resnet_modle.net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize the hyper_params
num_epochs = 30
lr = 0.00003

# initialize the params
for params in net.parameters():
    params_x = torch.Tensor(params)
    params_n = params_x.data.numpy()
    if params_n.ndim > 1:
        init.kaiming_normal_(params)
    else:
        init.normal_(params)

# define the loss
loss_function = nn.CrossEntropyLoss()

# do the training
# define the optimizer0

optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.0001,
                            momentum=0.9)
simple_function.train(net, train_iter, test_iter, loss_function, num_epochs,
                      optimizer=optimizer, device=device)

xpath = 'F:\\Awinter\\MCM\\C\\unprocessed_'
features_process = image(xpath)
features_process_tensor = torch.Tensor(features_process)
y = net(features_process_tensor)
print(y)
