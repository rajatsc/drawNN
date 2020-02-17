import torch.nn.functional as F

def cross_entropy_loss(output, target):
	"""
	The input is expected to contain scores for each class.
	input has to be a 2D Tensor of size (minibatch, C).
	This criterion expects a class index (0 to C-1) as the target 
	for each value of a 1D tensor of size minibatch


	Making it a 2D tensor still did not work for me. 
	CrossEntropyLoss takes a 1D tensor. I had to squeeze
	 the last dimension with target.squeeze(1) so it becomes a 1D tensor of size (batch, ).
	"""
	target = target.squeeze(1)
	return F.cross_entropy(output, target)