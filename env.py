#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 19:05:26 2023

@author: hungyunlu
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from torch.autograd import Variable
from torch.distributions import MultivariateNormal, Gamma

class NF(gym.Env):

    def __init__(self, params, render_mode=None):
        super().__init__()
        for key,value in params.items(): setattr(self,key,value)
        self.observation_space = spaces.Box(low=np.array([0,0]),high=np.array([self.max_beta, self.max_gamma]))
        self.action_space = spaces.Box(low=np.array([-np.inf,-np.inf]), high=np.array([np.inf,np.inf]))

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.beta       = Gamma(torch.tensor([2.167]),torch.tensor([1/4.437])).sample() #torch.rand(1)*120 # Initiate from anywhere
        self.gamma      = Gamma(torch.tensor([5.013]),torch.tensor([1/1.069])).sample() #torch.rand(1)*120 # Initiate from anywhere
        self.target     = 0.8
        self.holdtime   = 0
        self.complete   = False
        return self.observation, None

    def step(self, action):

        ## Exploration
        if np.random.random() > self.explore_thres:
            self.explore_discount = 0

        ## Calculate exploration discounted action
        action = [act*self.factor for act in action]
        self.explore_discount += 1

        ## Calculate pulling from increasing too much power
        action = [action[0]-self.pull[0], action[1]-self.pull[1]]

        ## Exploitation with original action
        self.beta = torch.clamp(self.beta + action[0], min=0.5,max=self.max_beta)
        self.gamma = torch.clamp(self.gamma + action[1], min=0.5,max=self.max_gamma)
        self.step_count += 1
        terminated,reward = False, - self.step_count / 1e8 - self.distance.item() / 1e8

        ## Calculate hold time
        if self.hold:
            self.holdtime += 1
        else:
            self.holdtime = 0

        ## Finishing a trial
        if self.holdtime > 12:# Greater than 200ms, sampling at 60Hz
            terminated,reward,self.complete = True,1,True

        ## Not finishing a trial
        truncated   = True if self.step_count > self.episode_cutoff else False

        return self.observation,reward,terminated,truncated,None

    def render(self):
        return

    @property
    def pull(self):
        b = self.beta - self.mean_beta
        g = self.gamma - self.mean_gamma
        return [b**self.factor_m*self.factor_c, g**self.factor_m*self.factor_c]

    @property
    def factor(self):
        return self.factor_a*np.exp(-self.factor_b*self.explore_discount)+1

    @property
    def pf(self):
        return self.beta / (self.beta+self.gamma)

    @property
    def observation(self):
        return torch.tensor([self.beta,self.gamma])

    @property
    def distance(self):
        return abs(self.pf - self.target)

    @property
    def hold(self):
        return False if self.distance > 0.042 else True
