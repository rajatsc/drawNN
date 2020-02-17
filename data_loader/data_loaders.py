from torchvision import datasets, transforms
from base import BaseDataLoader
from dataset import DrawNNDataset

class DrawNNDataLoader(BaseDataLoader):
	"""
	MNIST data loading demo using BaseDataLoader
	"""
	def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
		
		#transform is a list of transformation object



		transform = transforms.Compose([
			transforms.ToTensor(),
			#transforms.Normalize(mean = [0.1307], std = [0.3081])
		])
		self.data_dir = data_dir
		
		#initializing custom dataset
		self.dataset = DrawNNDataset(self.data_dir, train=training, transform=transform)
		super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)