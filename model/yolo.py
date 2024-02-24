import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNmodel(nn.Module):
  def __init__(self,input_channels,output_channels,kernel_size,stride,padding):
    super(CNNmodel,self).__init__()
    self.conv = nn.Conv2d(input_channels,output_channels,kernel_size,stride,padding)
    self.batchnorm = nn.BatchNorm2d(output_channels)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.conv(x)
    x = self.batchnorm(x)
    x = self.relu(x)
    return x

class DarkNet(nn.Module):
  def __init__(self,architecture_config,input_channels):
    super(DarkNet,self).__init__()
    self.architecture_config = architecture_config
    self.layers = []

    for layer in self.architecture_config:
      if type(layer) == tuple:
        kernel_size = layer[0]
        output_channels = layer[1]
        stride = layer[2]
        padding = layer[3]
        self.layers.append(
            CNNmodel(input_channels,output_channels,kernel_size,stride,padding)
        )
        input_channels = output_channels
      elif type(layer) == str:
        self.layers.append(
            nn.MaxPool2d(2,2)
        )
      elif type(layer) == list:
        layer1 = layer[0]
        layer2 = layer[1]
        repeat = layer[2]
        for _ in range(repeat):
          kernel_size = layer1[0]
          output_channels = layer1[1]
          stride = layer1[2]
          padding = layer1[3]
          self.layers.append(
            CNNmodel(input_channels,output_channels,kernel_size,stride,padding)
          )

          input_channels = output_channels
          kernel_size = layer2[0]
          output_channels = layer2[1]
          stride = layer2[2]
          padding = layer2[3]

          self.layers.append(
              CNNmodel(input_channels,output_channels,kernel_size,stride,padding)
          )
          input_channels = output_channels

    self.darknet_model =  nn.Sequential(*self.layers)

  def forward(self,x):
      return self.darknet_model(x)

class Yolov1(nn.Module):
  def __init__(self,input_channels,split_size,num_boxes,num_classes):
    super(Yolov1,self).__init__()
    self.architecture_config = model_architecture
    self.darknet = DarkNet(self.architecture_config,input_channels)
    self.fc1 = nn.Linear(1024*split_size*split_size,496)
    self.fc2 = nn.Linear(496,split_size*split_size * ( num_classes + (num_boxes*5)))
    self.flatten = nn.Flatten()
    self.leaky_relu = nn.LeakyReLU()
    self.dropout = nn.Dropout(0.5)

  def forward(self,x):
    x = self.darknet(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.dropout(x)
    x = self.leaky_relu(x)
    out_x = self.fc2(x)
    return out_x