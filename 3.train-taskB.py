#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import torch
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
from PIL import Image
from copy import deepcopy

from ewc import train_ewc
from config import data_params, nn_params, opt_params, earlystop_params, params
from models import NeuralNetwork, test, CheckPoint, EarlyStopping


#------------------------------------------------------------------------------
#	Permute rows of an image
#------------------------------------------------------------------------------
def permute_image(img, ind_permute):
	"""ind_permute: numpy ndarray, ind_permute shape: (28,)"""
	ind_permute = [14,27,24,1,23,0,9,12,10,17,16,22,13,11,
					8,25,2,5,21,4,26,20,19,7,3,15,18,6]
	img_new = np.array(img)[ind_permute, :]
	return Image.fromarray(img_new)

transform_permute = lambda img_batch: permute_image(img_batch, ind_permute)


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
# Dataset
trainB_loader = torch.utils.data.DataLoader(
	datasets.MNIST(
		'MNIST_data',
		train=True,
		download=True,
		transform=transforms.Compose([
			transforms.Lambda(transform_permute),
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))])),
	shuffle=False, **data_params)

testA_loader = torch.utils.data.DataLoader(
	datasets.MNIST(
		'MNIST_data',
		train=False,
		download=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))])),
	shuffle=False, **data_params)

testB_loader = torch.utils.data.DataLoader(
	datasets.MNIST(
		'MNIST_data',
		train=False,
		download=True,
		transform=transforms.Compose([
			transforms.Lambda(transform_permute),
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))])),
	shuffle=False, **data_params)


# Model
model = NeuralNetwork(**nn_params)
model = model.to(model.device)
model.load("modelA.ckpt")
prev_opt_thetas = deepcopy(list(model.parameters()))

optimizer = optim.Adam(model.parameters(), **opt_params)
base_loss_fn = torch.nn.CrossEntropyLoss(reduction="elementwise_mean")


# Generate permute indices
ind_permute = np.arange(0, 28)
np.random.shuffle(ind_permute)
np.save("permuteB.npy", ind_permute)


# Load the previous FIM
fishers_cpu = torch.load("fisherA.pth")
fishers = []
for fisher in fishers_cpu:
	fishers.append(fisher.to(model.device))


# Create callbacks
checkpoint = CheckPoint(model, "modelB.ckpt")
earlystop = EarlyStopping(**earlystop_params)


# Train and evaluate
flg_stop = False
for epoch in range(1, params["n_epochs"] + 1):
	print("\n[EPOCH %d]" % (epoch))
	loss_train = train_ewc(model, trainB_loader, optimizer, base_loss_fn,
						params["lamda"], fishers, prev_opt_thetas, epoch,
						description="Train on task B")
	print()
	loss_test, acc_test = test(model, testB_loader, base_loss_fn,
						description="Test on task B")
	print()
	test(model, testA_loader, base_loss_fn,
						description="Test on task A")
	print()

	# Callbacks
	checkpoint.backup(loss_test)
	flg_stop = earlystop.check(loss_test)
	if flg_stop:
		break
	print("------------------------------------------------------------------")