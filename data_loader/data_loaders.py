from torchvision import datasets, transforms
from base import BaseDataLoader
from dataset import myDataset

class myDataLoader(BaseDataLoader):
	"""
	MNIST data loading demo using BaseDataLoader
	"""
	def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
		
		#transform is a list of transformation object

		transform = transforms.Compose([
			transforms.ToTensor(), #ToTensor() scales the values between 0 to 1
			#transforms.Normalize(mean = [0.1307], std = [0.3081])
		])
		self.data_dir = data_dir
		
		#initializing custom dataset
		self.dataset = myDataset(self.data_dir, train=training, transform=transform)
		super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)