#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

from models import NeuralNetwork
from ewc import compute_fisher
from config import data_params, nn_params


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
# Dataset
trainA_loader = torch.utils.data.DataLoader(
	datasets.MNIST(
		'MNIST_data',
		train=True,
		download=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])),
	shuffle=True, **data_params)


# Model
model = NeuralNetwork(**nn_params)
model = model.to(model.device)
model.load("modelA.ckpt")


# Compute and save Fisher Information matrix
for X, Y in trainA_loader:
	pass
X = X.view(-1, model.input_size).to(model.device)
Y = Y.to(model.device)
fishers = compute_fisher(model, X, Y)
for fim in fishers:
	print(fim)
torch.save(fishers, "fisherA.pth")


# # Visualize the diff_means
# plt.figure(1)
# plt.plot(diff_means)
# plt.grid(True)
# plt.show()