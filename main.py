#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 19:06:36 2023

@author: hungyunlu
"""

from utils import ValueNet, ActorNet, Logger, make_pf, slide_avg
from env import NF
from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

def pick_sample(s):
    with torch.no_grad():
        result = actor_func(torch.tensor([s]))
        mean, log_std, off_diag = result[:2],result[2:4],result[4]
        std = torch.exp(log_std)
        L = torch.tensor([[std[0],0],[off_diag,std[1]]])
        m = MultivariateNormal(mean, torch.matmul(L,torch.transpose(L,0,1)))
        action = m.sample()
        return action.tolist()

parameters = {
    'max_beta':120,
    'max_gamma':120,
    'mean_beta':2.167*4.437,
    'mean_gamma':5.013*1.069,
    'episode_cutoff':600,
    'explore_thres':0.99,
    'explore_discount':np.inf,
    'factor_a':15,
    'factor_b':0.15,
    'factor_c':1e-4,
    'factor_m':2
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

value_func = ValueNet().to(device)
actor_func = ActorNet().to(device)
value_optimizer = torch.optim.Adam(value_func.parameters(), lr=0.0001) # Separate optimizers
actor_optimizer = torch.optim.Adam(actor_func.parameters(), lr=0.0001) # Separate optimizers
gamma = 0.96 # Reward discount rate
env   = NF(parameters)
log   = Logger()
reward_records = []
for epi in range(3001):

    ## Start a trial
    done = False
    trajectory,actions,rewards,factors = [],[],[],[]
    s, _ = env.reset()
    while not done:
        trajectory.append(s.tolist())
        a = pick_sample(env.pf)
        s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        actions.append(a)
        rewards.append(r)
        factors.append(env.factor)

    ## Get infinite-horizon cumulative rewards
    cum_rewards = np.zeros_like(rewards)
    reward_len = len(rewards)
    for j in reversed(range(reward_len)):
        cum_rewards[j] = rewards[j] + (cum_rewards[j+1]*gamma if j+1 < reward_len else 0)

    ## Optimize value loss (Critic)
    value_optimizer.zero_grad()
    states = torch.tensor(trajectory, dtype=torch.float).to(device)
    cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)
    values  = value_func(states)
    values  = values.squeeze(dim=1)
    vf_loss = F.smooth_l1_loss(values,cum_rewards,reduction="mean")
    vf_loss.backward()
    nn.utils.clip_grad_norm_(value_func.parameters(),5)
    value_optimizer.step()

    ## Optimize policy loss (Actor)
    with torch.no_grad(): # Very important - gradient doesn't flow through the critic network
        values = value_func(states)
    actor_optimizer.zero_grad()
    actions = torch.tensor(actions, dtype=torch.float).to(device)
    advantages = cum_rewards - values.squeeze(1)
    result = actor_func(make_pf(states).unsqueeze(1))
    mean, log_std, off_diag = result[:,:2],result[:,2:4],result[:,4]
    std = torch.exp(log_std)
    L = torch.zeros(std.size(0),2,2)
    L[:,0,0] = std[:,0]
    L[:,1,1] = std[:,1]
    L[:,1,0] = off_diag
    m = MultivariateNormal(mean, scale_tril=L)
    log_probs = m.log_prob(actions)
    pi_loss = - log_probs * advantages
    pi_loss.sum().backward()
    nn.utils.clip_grad_norm_(actor_func.parameters(),0.5)
    actor_optimizer.step()

    ## Calculate state values
    if not epi%200:
        value_map = np.zeros((120,120))
        for i in range(120):
            for j in range(120):
                value_map[i,j] = value_func.v_net(torch.tensor([i,j],dtype=torch.float32))
        value_map = (value_map-value_map.min())/(value_map.max()-value_map.min())
    else: value_map = None

    ## Logging
    log.write(reward      = sum(rewards),
              actor_loss  = pi_loss.sum().item(),
              critic_loss = vf_loss.item(),
              trajectory  = trajectory,
              value       = value_map,
              complete    = env.complete,
              n_steps     = env.step_count,
              factor      = factors)

    ## Report and stop if needed
    if epi>50:
        mean_reward = np.mean(log.reward_records[-50:])
        print(f"Run episode {epi} with reward {mean_reward:.2f}", end="\r")
        if mean_reward >= 0.85:
            break
env.close()


savefig = False
fig_type = 'svg'
model_id = '20230809_v1'

print(parameters)

###########
# Figure 1.
# State value maps
###########
state_values = [i for i in log.state_values if i is not None]
plt.figure(figsize=(10,10))
for x in range(len(state_values)):
    plt.subplot(4,4,x+1)
    plt.pcolormesh(state_values[x],vmin=0,vmax=1,cmap='jet')
    # plt.colorbar(label='Normalized state value')
    # plt.xlabel('Gamma power')
    # plt.ylabel('Beta power')
    plt.title(f'State value map {x*200}')
plt.subplots_adjust(hspace=0.3,wspace=0.3)
if savefig: plt.savefig(f'Data/model_performance/{model_id}_state_values.png')
plt.show()
###########
# Figure 2.
# Cumulative reward
# actor/critic losses
#
###########

plt.figure(figsize=(5,8))
ax = plt.subplot(411)
ax.plot(slide_avg(log.reward_records,50),c='k')
ax.set_ylabel('Avg reward')
ax1 = ax.twinx()
ax1.eventplot(np.where(log.complete)[0],color='g')
ax1.set_yticks([])
ax1.set_ylim([0,10])
ax.set_xlabel('Episode number')
ax2 = plt.subplot(412)
ax2.set_xlabel('Episode number')
ax3 = ax2.twinx()
ax2.plot(slide_avg(log.critic_losses,100),c='r',label='Critic loss')
ax3.plot(-slide_avg(log.actor_losses,100),c='b',label='Actor loss')
ax2.legend(loc ='upper center',frameon=False)
ax3.legend(loc ='upper right',frameon=False)
ax2.tick_params(axis='y', colors='r')
ax3.tick_params(axis='y', colors='b')
ax2.set_ylabel('Losses')
x = []
for i in range(len(log.factor)): x.append(sum(np.array(log.factor[i])==max(log.factor[0])))
plt.subplot(413)
plt.plot(slide_avg(x,20),c='k')
plt.ylabel('Slide avg # attempts')
plt.xlabel('Episode number')
plt.subplot(414)
plt.hist(np.array(x)-0.5,bins=max(x),ec='k',fc='none')
plt.xticks(range(max(x)))
plt.xlabel('Number of attempts')
plt.ylabel('Count')
plt.subplots_adjust(hspace=0.4)
if savefig: plt.savefig(f'Data/model_performance/{model_id}_performance.{fig_type}')
plt.show()

###########
# Figure 3.
# Trajectories
###########
def plot_states(trajectory):
    cmap = plt.cm.jet_r
    states = np.array(trajectory)
    x,y = slide_avg(states[:,1],10),slide_avg(states[:,0],10)
    if len(x) == 601: cmap,alpha = plt.cm.binary,0.2
    else: cmap,alpha = plt.cm.jet, 1
    z = np.arange(len(x)) / 600
    plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], z[1:],scale_units='xy', angles='xy', scale=1,cmap=cmap,alpha=alpha)

n = 100
plt.figure(figsize=(9,4))
plt.subplot(121)
for i in range(n):
    plot_states(log.trajectories[i])
plt.xlim([-2,122])
plt.ylim([-2,122])
plt.xlabel('Gamma power')
plt.ylabel('Beta power')
plt.title(f'First {np.sum(log.complete[:n])}/{n} complete trials')
plt.subplot(122)
for i in range(n):
    plot_states(log.trajectories[-i])
# plt.colorbar(label='Time stamp')
plt.xlim([-2,122])
plt.ylim([-2,122])
plt.xlabel('Gamma power')
plt.ylabel('Beta power')
plt.title(f'Last {np.sum(log.complete[-n:])}/{n} complete trials')
if savefig: plt.savefig(f'Data/model_performance/{model_id}_trajectory.{fig_type}')
plt.show()
###########
# Figure 4.
# Number of steps for each complete episodes
###########
y = slide_avg(np.array(log.n_steps)[log.complete],15)
_,p_steps = stats.ttest_ind(y[:50],y[-50:])
plt.plot(y,c='k')
plt.ylabel('Steps to complete an episode')
plt.xlabel('Complete episode #')
if savefig: plt.savefig(f'Data/model_performance/{model_id}_steps.{fig_type}')
plt.show()
###########
# Figure 5.
# Compare first and last 50 episodes
###########
plt.figure(figsize=(7,3))
plt.subplot(121)
x = slide_avg(log.reward_records,50)
_,p_reward = stats.ttest_ind(x[:50],x[-50:])

plt.bar(-0.2,np.mean(x[:50]),width=0.4,yerr=np.std(x[:50]),capsize=5)
plt.bar(0.2,np.mean(x[-50:]),width=0.4,yerr=np.std(x[-50:]),capsize=5)
plt.ylabel('Avg reward')
plt.xticks([-0.2,0.2],['First 50', 'Last 50'])
plt.subplot(122)
plt.bar(-0.2,np.mean(y[:50]),width=0.4,yerr=np.std(y[:50]),capsize=5)
plt.bar(0.2,np.mean(y[-50:]),width=0.4,yerr=np.std(y[-50:]),capsize=5)
plt.ylabel('Avg steps')
plt.xticks([-0.2,0.2],['First 50', 'Last 50'])
plt.subplots_adjust(wspace=0.4)
if savefig: plt.savefig(f'Data/model_performance/{model_id}_firstlast_compare.{fig_type}')
plt.show()
print(f'P-values - Rewards: {p_reward}, Steps {p_steps}')
