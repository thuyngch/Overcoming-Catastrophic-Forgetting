#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import numpy as np
from tqdm import tqdm
from functools import reduce
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


#------------------------------------------------------------------------------
#	Model
#------------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
	def __init__(self, input_size, output_size, hidden_num, hidden_size,
				input_dropout, hidden_dropout, map_location):
		# Attributes
		super(NeuralNetwork, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_num = hidden_num
		self.hidden_size = hidden_size
		self.input_dropout = input_dropout
		self.hidden_dropout = hidden_dropout
		self.map_location = map_location

		# Architecture
		self.layers = nn.ModuleList([
            # Input layer
            nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
            nn.Dropout(self.input_dropout),

            # Hidden layers
            *((nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
               nn.Dropout(self.hidden_dropout)) * self.hidden_num),

            # Output layer
            nn.Linear(self.hidden_size, self.output_size)
		])

	@property
	def device(self):
		return torch.device(self.map_location)

	def forward(self, X):
		return reduce(lambda x, l: l(x), self.layers, X.view(-1, self.input_size))

	def load(self, model_file):
		trained_dict = torch.load(model_file, map_location=self.map_location)
		self.load_state_dict(trained_dict)


#------------------------------------------------------------------------------
#	Train the model within an epoch
#------------------------------------------------------------------------------
def train(model, train_loader, optimizer, loss_fn, epoch, description=""):
	model.train()
	loss_train = 0
	pbar = tqdm(train_loader)
	pbar.set_description(description)
	for inputs, targets in pbar:
		inputs, targets = inputs.to(model.device), targets.to(model.device)

		optimizer.zero_grad()
		logits = model(inputs)
		loss = loss_fn(logits, targets)
		loss_train += loss.item()
		loss.backward()
		optimizer.step()

	loss_train /= len(train_loader)
	print('loss_train: {:.6f}'.format(loss_train))
	return loss_train


#------------------------------------------------------------------------------
#	Test the model
#------------------------------------------------------------------------------
def test(model, test_loader, loss_fn, description=""):
	model.eval()
	loss_test, correct = 0, 0
	with torch.no_grad():
		pbar = tqdm(test_loader)
		pbar.set_description(description)
		for inputs, targets in pbar:
			inputs, targets = inputs.to(model.device), targets.to(model.device)
			logits = model(inputs)
			loss_test += loss_fn(logits, targets).item()
			pred = logits.max(1, keepdim=True)[1]
			correct += pred.eq(targets.view_as(pred)).sum().item()

	loss_test /= len(test_loader)
	acc_test = 100. * correct / len(test_loader.dataset)
	print('loss_test: {:.6f}, acc_test: {:.2f}%'.format(loss_test, acc_test))
	return loss_test, acc_test


#------------------------------------------------------------------------------
#	Checkpoint
#------------------------------------------------------------------------------
class CheckPoint(object):
	def __init__(self, model, model_file):
		super(CheckPoint, self).__init__()
		self.model = model
		self.model_file = model_file
		self.best_loss_valid = np.inf

	def backup(self, loss_valid):
		if loss_valid < self.best_loss_valid:
			print("Validation loss improved from %.4f to %.4f" %
				(self.best_loss_valid, loss_valid))
			torch.save(self.model.state_dict(), self.model_file)
			print("Model is saved in", self.model_file)
			self.best_loss_valid = loss_valid
		else:
			print("Validation loss is not improved")


#------------------------------------------------------------------------------
#	Early Stopping
#------------------------------------------------------------------------------
class EarlyStopping(object):
	def __init__(self, num_bad_epochs=1, improved_thres=0.01):
		super(EarlyStopping, self).__init__()
		self.num_bad_epochs = num_bad_epochs
		self.improved_thres = improved_thres

		self.not_improved = 0
		self.best_loss_valid = np.inf

	def check(self, loss_valid):
		if (self.best_loss_valid-loss_valid)>=self.improved_thres:
			self.not_improved = 0
		else:
			self.not_improved += 1
			if self.not_improved==self.num_bad_epochs:
				print("Early stopping")
				return True

		if loss_valid < self.best_loss_valid:
			self.best_loss_valid = loss_valid

		return False


#------------------------------------------------------------------------------
#	Logger
#------------------------------------------------------------------------------
class Logger(object):
	def __init__(self, list_metrics, logger_file="log.npy"):
		super(Logger, self).__init__()
		self.logger_file = logger_file

		self.metrics = {}
		for metric_name in list_metrics:
			self.metrics[metric_name] = []


	def update(self, **kwargs):
		for key, value in kwargs.items():
			self.metrics[key].append(value)
		np.save(self.logger_file, self.metrics)


	def load(self):
		self.metrics = np.load(self.logger_file).item()


	def visualize(self, fg_idx=1):
		plt.figure(fg_idx)
		metric_names = list(self.metrics.keys())

		plt.subplot(1,2,1); plt.title("Loss")
		legends = []
		for metric_name in metric_names:
			if not "loss" in metric_name: continue
			metric_val = self.metrics[metric_name]
			n_samples = len(metric_val)
			plt.plot(range(1, n_samples+1), metric_val)
			legends.append(metric_name)
		plt.legend(legends); plt.grid(True)
		plt.xlabel("epoch"); plt.ylabel("loss")

		plt.subplot(1,2,2); plt.title("Accuracy")
		legends = []
		for metric_name in metric_names:
			if not "acc" in metric_name: continue
			metric_val = self.metrics[metric_name]
			n_samples = len(metric_val)
			plt.plot(range(1, n_samples+1), metric_val)
			legends.append(metric_name)
		plt.legend(legends); plt.grid(True)
		plt.xlabel("epoch"); plt.ylabel("accuracy")
		plt.show()