import os
import torch
import torch.utils.data as data
import numpy as np

class myDataset(data.Dataset):

	def __init__(self, data_dir, train, transform=None):
		self.classes = ["airplane", "apple", "axe", "bed", "bicycle", "boomerang", "car", "ear", "eye", "fan", "flower",
		   "helicopter", "hexagon", "house", "ice cream", "mouse", "mushroom", "octopus", "shark", "spoon",
		   "star", "triangle", "t-shirt", "umbrella", "zigzag"]

		self.transform = transform
		self.train = train
		self.data_dir = data_dir
		self.num_classes = len(self.classes)

		if self.train:
			self.samples = 10000
		else:
			self.samples = 2500




	def __len__(self):
		return self.samples*self.num_classes



	def __getitem__(self, index):
		
		"""
		In PyTorch, images are represented as 
		[channels, height, width], so a color image would be.
	
		During the training you will get batches of images, 
		so your shape in the forward method will get an
		additional batch dimension at dim0: [batch_size, channels, height, width]

		"""

		class_index = int(index/self.samples)
		file = "full_numpy_bitmap_{}.npy".format(self.classes[class_index])
		full_file_path = os.path.join(self.data_dir, file)

		if self.train:
			sample_index = index%self.samples
		else:
			sample_index = self.samples + index%self.samples
		data = np.load(full_file_path)[sample_index].reshape(28, 28, 1)
		
		numpy_label = np.array([class_index])
		label = torch.from_numpy(numpy_label)
		
		if self.transform:
			data = self.transform(data)

		return (data, label)
