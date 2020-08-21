## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.conv1_batch_norm = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2_batch_norm = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv3_batch_norm = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv4_batch_norm = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dense1 = nn.Linear(12*12*256, 2048)
        self.dense1_batch_norm = nn.BatchNorm1d(2048)
        self.dense2 = nn.Linear(2048, 1028)
        self.dense2_batch_norm = nn.BatchNorm1d(1028)
        self.dense3 = nn.Linear(1028,136)
        self.drop = nn.Dropout(0.25)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv1_batch_norm(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_batch_norm(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_batch_norm(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_batch_norm(self.conv4(x))))
        
        x = x.view(x.size(0),-1)
        
        x = self.drop(F.relu(self.dense1_batch_norm(self.dense1(x))))
        x = self.drop(F.relu(self.dense2_batch_norm(self.dense2(x))))
        x = self.dense3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x