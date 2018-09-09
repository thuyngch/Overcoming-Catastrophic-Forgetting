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
from models import NeuralNetwork, test, CheckPoint, EarlyStopping, Logger


#------------------------------------------------------------------------------
#	Permute rows of an image
#------------------------------------------------------------------------------
def permute_image(img, ind_permute):
	"""ind_permute: numpy ndarray, ind_permute shape: (28,)"""
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
np.random.seed(0)
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
list_metrics = ["loss_trainB", "loss_testB", "acc_testA", "acc_testB"]
logger = Logger(list_metrics=list_metrics, logger_file="log-metrics.npy")


# Train and evaluate
flg_stop = False
for epoch in range(1, params["n_epochs"] + 1):
	print("\n[EPOCH %d]" % (epoch))
	loss_trainB = train_ewc(model, trainB_loader, optimizer, base_loss_fn,
						params["lamda"], fishers, prev_opt_thetas, epoch,
						description="Train on task B")
	print()
	loss_testB, acc_testB = test(model, testB_loader, base_loss_fn,
						description="Test on task B")
	print()
	_, acc_testA = test(model, testA_loader, base_loss_fn,
						description="Test on task A")
	print()

	# Callbacks
	checkpoint.backup(loss_testB)
	flg_stop = earlystop.check(loss_testB)
	logger.update(loss_trainB=loss_trainB, loss_testB=loss_testB,
					acc_testA=acc_testA, acc_testB=acc_testB)
	if flg_stop:
		break
	print("------------------------------------------------------------------")


# Visualize the training progress
logger.visualize(fg_idx=1)