## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        layer_size_conv = [24, 48, 144]
        layer_size_lear = [2048]
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, layer_size_conv[0], 5, stride=2)  # (feature_1, 220, 220)
        I.xavier_normal_(self.conv1.weight, gain=1.)
        self.norm1 = nn.BatchNorm2d(layer_size_conv[0])
#         self.pool1 = nn.MaxPool2d(2, 2)  # (features_1, 110, 110)
        
        self.conv2 = nn.Conv2d(layer_size_conv[0], layer_size_conv[1], 3, stride=2)  # (features_2, 54, 54)
        I.xavier_normal_(self.conv2.weight, gain=1.)
        self.norm2 = nn.BatchNorm2d(layer_size_conv[1])
#         self.pool2 = nn.MaxPool2d(2, 2)  # (features_2, 27, 27)
        
        self.conv3 = nn.Conv2d(layer_size_conv[1], layer_size_conv[2], 3, stride=2)  # (features_3, 26, 26)
        I.xavier_normal_(self.conv3.weight, gain=1.)
        self.norm3 = nn.BatchNorm2d(layer_size_conv[2])
        self.pool3 = nn.MaxPool2d(2, 2)  # (features_3, 13, 13)
        
        self.fc1 = nn.Linear(layer_size_conv[2]*13*13, layer_size_lear[0])
        I.xavier_normal_(self.fc1.weight, gain=1.)
        self.nf1 = nn.BatchNorm1d(layer_size_lear[0])
        self.df1 = nn.Dropout(p=0.2)
        
#         self.fc2 = nn.Linear(layer_size_lear[0], layer_size_lear[1])
#         I.xavier_normal_(self.fc2.weight, gain=1.)
#         self.nf2 = nn.BatchNorm1d(layer_size_lear[1])
#         self.df2 = nn.Dropout(p=0.2)
        
        self.output = nn.Linear(layer_size_lear[-1], 136)
        
        ## Note that among the layers to add, consider including:
            # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.pool3(F.relu(self.norm3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
#         x = torch.flatten(x, 1)  # the future
        
        x = F.relu(self.df1(self.nf1(self.fc1(x))))
#         x = self.df2(F.relu(self.nf2(self.fc2(x))))
        
        x = self.output(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    
class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))