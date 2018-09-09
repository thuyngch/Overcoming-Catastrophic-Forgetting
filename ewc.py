#------------------------------------------------------------------------------
#    Libraries
#------------------------------------------------------------------------------
import torch
from torch import autograd
import torch.nn.functional as F

from tqdm import tqdm


#------------------------------------------------------------------------------
#	Compute Fisher Information Matrix (FIM)
#------------------------------------------------------------------------------
def compute_fisher(model, X, Y):
	# Instantiate the FIM
	fishers = []
	n_samples = X.shape[0]
	for param in model.parameters():
		fishers.append(torch.zeros_like(param))

	# # Compute the FIM (get mean of gradients)
	# diff_means = [0.] * (n_samples-1)
	# logits = model(X)
	# loglikelihoods = F.log_softmax(logits, dim=1)[range(n_samples), Y]
	# for i in range(n_samples):
	# 	loglikelihood = loglikelihoods[i]
	# 	loglikelihood.backward(retain_graph=True)
	# 	for idx, param in enumerate(model.parameters()):
	# 		if i>0:
	# 			delta = (param.grad**2)/(i+1) - fishers[idx]/(i*(i+1))
	# 			diff_means[i-1] += torch.abs(delta).sum()
	# 		fishers[idx] += param.grad**2
	# for idx, param in enumerate(model.parameters()):
	# 	fishers[idx] /= n_samples
	# return fishers, diff_means

	# Compute the FIM (get mean of loglikelihoods)
	logits = model(X)
	loglikelihoods = F.log_softmax(logits, dim=1)[range(n_samples), Y]
	loglikelihood = loglikelihoods.mean()
	loglikelihood.backward()
	for idx, param in enumerate(model.parameters()):
		fishers[idx] += param.grad**2
	return fishers


#------------------------------------------------------------------------------
#    Cusstom loss funtion
#------------------------------------------------------------------------------
def ewc_loss(logits, targets, lamda, fishers, prev_opt_thetas, cur_thetas):
	loss = 0
	for i in range(len(fishers)):
		fisher = fishers[i]
		prev_opt_theta = prev_opt_thetas[i]
		cur_theta = cur_thetas[i]
		delta = ((prev_opt_theta-cur_theta)**2).sum()
		loss += lamda/2 * torch.sum(fisher * (prev_opt_theta-cur_theta)**2)
	return loss


#------------------------------------------------------------------------------
#	Training process using EWC method
#------------------------------------------------------------------------------
def train_ewc(model, train_loader, optimizer, base_loss_fn,
			lamda, fishers, prev_opt_thetas, epoch, description=""):
	model.train()
	loss_train = 0
	pbar = tqdm(train_loader)
	pbar.set_description(description)
	for inputs, targets in pbar:
		inputs, targets = inputs.to(model.device), targets.to(model.device)
		cur_thetas = list(model.parameters())

		optimizer.zero_grad()
		logits = model(inputs)
		loss_crossentropy = base_loss_fn(logits, targets)
		loss_ewc = ewc_loss(logits, targets, lamda, fishers,
							prev_opt_thetas, cur_thetas)

		loss = loss_crossentropy + loss_ewc
		loss_train += loss.item()
		loss.backward()
		optimizer.step()

	loss_train /= len(train_loader)
	print('loss_train: {:.6f}'.format(loss_train))
	return loss_train