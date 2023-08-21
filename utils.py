#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 19:05:16 2023

@author: hungyunlu
"""
import numpy as np
import torch.nn as nn


def make_pf(s):
    return 1/(1+s[:,1]/s[:,0])

def slide_avg(array, n):
    avg = np.zeros(len(array))
    for i in range(len(array)):
        if i < n:
            avg[i] = np.sum(array[:i+1])/float(i+1)
        else:
            avg[i] = np.sum(array[i-n+1:i+1])/float(n)
    return avg

class Logger:
    def __init__(self):
        self.reward_records = []
        self.state_values   = []
        self.actor_losses   = []
        self.critic_losses  = []
        self.trajectories   = []
        self.complete       = []
        self.n_steps        = []
        self.factor         = []
    def write(self,reward,value,actor_loss,critic_loss,trajectory,complete,n_steps,factor):
        self.reward_records .append(reward)
        self.state_values   .append(value)
        self.actor_losses   .append(actor_loss)
        self.critic_losses  .append(critic_loss)
        self.trajectories   .append(trajectory)
        self.complete       .append(complete)
        self.n_steps        .append(n_steps)
        self.factor         .append(factor)

class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi_net = nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.Sigmoid(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,5))

    def forward(self, s):
        return self.pi_net(s)

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(2,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.Sigmoid(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid())

    def forward(self, s):
        return self.v_net(s)
