#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from config import data_params, nn_params, opt_params, earlystop_params, params
from models import NeuralNetwork, train, test, CheckPoint, EarlyStopping


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
			transforms.Normalize((0.1307,), (0.3081,))])),
	shuffle=True, **data_params)

testA_loader = torch.utils.data.DataLoader(
	datasets.MNIST(
		'MNIST_data',
		train=False,
		download=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))])),
	shuffle=False, **data_params)


# Model
model = NeuralNetwork(**nn_params)
model = model.to(model.device)
optimizer = optim.Adam(model.parameters(), **opt_params)
loss_fn = torch.nn.CrossEntropyLoss(reduction="elementwise_mean")


# Create callbacks
checkpoint = CheckPoint(model, "modelA.ckpt")
earlystop = EarlyStopping(**earlystop_params)


# Train and evaluate the model
flg_stop = False
for epoch in range(1, params["n_epochs"] + 1):
	print("\n[EPOCH %d]" % (epoch))
	loss_train = train(model, trainA_loader, optimizer, loss_fn, epoch,
						description="Train on task A")
	print()
	loss_test, acc_test = test(model, testA_loader, loss_fn,
						description="Test on task A")
	print()

	# Callbacks
	checkpoint.backup(loss_test)
	flg_stop = earlystop.check(loss_test)
	if flg_stop:
		break
	print("------------------------------------------------------------------")