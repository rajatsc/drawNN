import torch
import torch.nn as nn
import math
import numpy as np
from base import BaseModel


class mySimpleModel(BaseModel):
	def __init__ (self, input_size=28, num_classes = 25):
		super().__init__()
		self.num_classes = num_classes

		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False), 
			nn.ReLU(inplace=True), 
			nn.MaxPool2d(2, 2)
			)

		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
			nn.ReLU(inplace=True), 
			nn.MaxPool2d(2, 2)
			)

		dimension = int(16 * math.pow(input_size/4, 2))
		#self.fc1 = nn.Sequential(nn.Linear(dimension, 512), nn.Dropout(0.5))
		#self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.Dropout(0.5))
		self.fc1 = nn.Sequential(nn.Linear(dimension, num_classes))


	def forward(self, input):
		#print(input.shape)
		output = self.conv1(input)
		output = self.conv2(output)
		output = output.view(output.size(0), -1)
		output = self.fc1(output)
		#output = self.fc2(output)
		#output = self.fc3(output)
		return output