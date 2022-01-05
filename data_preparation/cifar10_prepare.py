#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np

import os
import argparse
import time
from tqdm import tqdm

import torchvision.models as models
from resnet import ResNet18, ResNet50
from data_prepare import *
from copy import deepcopy

import matplotlib.pyplot as plt

batch_size = 128
device = "cuda"
data = "cifar10"
model = "resnet18"

def split_train_val(trainset, ratio=0.8):
    np.random.seed(1)
    randperm = np.random.permutation(len(trainset))
    np.random.seed(int(time.time()))
    train_indices = randperm[:int(len(randperm) * ratio)]
    test_indices = randperm[int(len(randperm) * ratio):]
    testset = deepcopy(trainset)
    def select_dataset(dataset, indices):
        dataset.targets, dataset.data = np.asarray(dataset.targets)[indices], np.asarray(dataset.data)[indices]
        return dataset
    trainset, testset = select_dataset(trainset, train_indices), select_dataset(testset, test_indices)
    testset.transform = transform_test
    return trainset, testset

def test(dataloader):
    net.eval()
    correct = 0
    total = 0
    logits = torch.zeros(len(dataloader.dataset), num_classes)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            
            logits[(batch_idx * batch_size):((batch_idx + 1) * batch_size)] = outputs
            
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            
            correct += predicted.eq(targets).sum().item()
    return logits

def get_conf_mat(logits, labels, num_classes):
    conf_mat = torch.zeros(num_classes, num_classes)
    for pred, label in tqdm(zip(logits.max(1)[1], labels)):
        conf_mat[label, pred] += 1
    conf_mat = (conf_mat.t() / conf_mat.sum(1)).t()
    return conf_mat

def T_scaling(logits, temperature):
    return torch.div(logits, temperature)


def _eval():
    loss = criterion(T_scaling(val_logits[cal_id].to(device), temperature.to(device)), val_targets[cal_id].to(device))
    loss.backward()
    return loss

#initiate dataset
print("Load cifar10 dataset")
transform_train = cifar_transform_train
transform_test = cifar_transform_test
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
num_classes = 10
input_size =  32
ratio = 0.6
trainset, valset = split_train_val(trainset, ratio=ratio)
testset.targets = np.asarray(testset.targets)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


#load model
print("Load the pretrained resnet18 model")
linear_base = input_size * input_size / 2
net = ResNet18(num_classes=num_classes, linear_base=linear_base).to(device)
checkpoint = torch.load(f"{data}_{model}.pth")
net.load_state_dict(checkpoint["net"])

print("compute the logits and the confusion ")
val_logits = test(valloader)
test_logits = test(testloader)

train_targets = torch.from_numpy(np.asarray(trainset.targets))
val_targets = torch.from_numpy(np.asarray(valset.targets))
test_targets = torch.from_numpy(np.asarray(testset.targets)) 

val_conf_mat = get_conf_mat(val_logits, val_targets, num_classes)
test_conf_mat = get_conf_mat(test_logits, test_targets, num_classes)

print("temperature calibrate the probability vector")
cal_num = int(1./4 * len(valset))
np.random.seed(0)
randperm = np.random.permutation(len(val_logits))
cal_id, other_id = randperm[:cal_num], randperm[cal_num:]

temperature = nn.Parameter(torch.ones(1).cuda())
criterion = nn.CrossEntropyLoss()
optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')
for i in range(100):
    optimizer.zero_grad()
    optimizer.step(_eval)

print("saving")
checkpoint = {}
checkpoint["val_preds"] = T_scaling(val_logits[other_id], temperature.cpu()).detach().numpy()
checkpoint["val_y"] = val_targets[other_id].numpy()
checkpoint["conf_mat"] = val_conf_mat
torch.save(checkpoint, "val_{}_{}_cal.pt".format(data, model))

p_train = np.bincount(train_targets.numpy()) / np.bincount(train_targets.numpy()).sum()
checkpoint = {}
checkpoint["test_preds"] = T_scaling(test_logits, temperature.cpu()).detach().numpy()
checkpoint["y"] = test_targets.numpy()
checkpoint["p_train"] = p_train
torch.save(checkpoint, "test_{}_{}_cal.pt".format(data, model))

