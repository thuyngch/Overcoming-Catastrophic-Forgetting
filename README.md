# Overcoming-Catastrophic-Forgetting
An implementation of the paper [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796) (DeepMind, 2016), using Pytorch framework.


#### Keywords: Neural Network, Catastrophic forgetting, Supervised learning, Pytorch


Table of contents
=================
- [What is the problem?](#what-is-the-problem)
- [How does the paper solve the problem?](#how-does-the-paper-solve-the-problem)
- [Project description](#project-description)
- [Result](#result)


# What is the problem?
Given several tasks, accompanying with their own datasets, we want to build a model of neural network to learn all of the tasks. Normally, we train the model with individual task in turn, in which, weights of the neural network are inherited from the last trained task. This, however, will lead to a phenomenon, called ***catastrophic forgetting***. Concretely, during the learning progress, weights of the neural network are updated in order to fit to the specific dataset. Therefore, knowledge that is learned from the previous task is likely to be forgotten in the subsequent task.


# How does the paper solve the problem?
By calculating Fisher Information Matrix (FIM), one can turn out how much parameters of a neural network are crucial to a task (dataset). Thus, authors of the paper modify the loss function, integrated FIM, to overcome the catastrophic forgetting. This configuration is named as Elastic Weight Consolidation (EWC). Particularly, weights, which are important to a previous task, are slowed down their update in a subsequent task and vice versa.


# Project description
My work is to implement the Superviced-learning experiment of the paper. Files, *"1.train-taskA.py"*, *"2.compute-fisherA.py"*, and *"3.train-taskB.py"*, are main executions. Meanwhile, utility functions are stored in files *"models.py"* and *"ewc.py"*. Besides, file *"config.py"* is used to configure parameters for the learning progress.


# Result

**Metrics of the last epoch, using conventional transfer learning configuration**
<p align="center">
  <img src="https://github.com/AntiAegis/Overcoming-Catastrophic-Forgetting/blob/raw_code/pics/normal-last-epoch.png" width="700" alt="accessibility text">
</p>

**Metrics of the last epoch, using EWC configuration**
<p align="center">
  <img src="https://github.com/AntiAegis/Overcoming-Catastrophic-Forgetting/blob/raw_code/pics/ewc-last-epoch.png" width="700" alt="accessibility text">
</p>

**Learning progress, using conventional transfer learning configuration**
<p align="center">
  <img src="https://github.com/AntiAegis/Overcoming-Catastrophic-Forgetting/blob/raw_code/pics/normal-plot.png" width="1000" alt="accessibility text">
</p>

**Learning progress, using EWC configuration**
<p align="center">
  <img src="https://github.com/AntiAegis/Overcoming-Catastrophic-Forgetting/blob/raw_code/pics/ewc-plot.png" width="1000" alt="accessibility text">
</p>
