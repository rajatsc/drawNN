import torch
import torch.nn as nn
import math
import numpy as np




class drawNNModel(nn.Module):
	def __init__ (self, input_size=28, num_classes = 25):
		super(drawNNModel, self).__init__()
		self.num_classes = num_classes
		self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
		self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
		dimension = int(64 * math.pow(input_size/4 - 3, 2))
		self.fc1 = nn.Sequential(nn.Linear(dimension, 512), nn.Dropout(0.5))
		self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.Dropout(0.5))
		self.fc3 = nn.Sequential(nn.Linear(128, num_classes))


	def forward(self, input):
		#print(input.shape)
		output = self.conv1(input)
		output = self.conv2(output)
		output = output.view(output.size(0), -1)
		output = self.fc1(output)
		output = self.fc2(output)
		output = self.fc3(output)
		return output
		
	def __str__(self):
		"""
		Model prints with number of trainable parameters
		"""
		model_parameters = filter(lambda p: p.requires_grad, self.parameters())
		params = sum([np.prod(p.size()) for p in model_parameters])
		return super().__str__() + '\nTrainable parameters: {}'.format(params)