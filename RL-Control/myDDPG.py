# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:57:22 2023

@author: Hanbz
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
# import matplotlib.pyplot as plt
# from matplotlib import cm

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage)-5, size=batch_size)
        s, a, r, s_ = [], [], [], []

        for i in ind:
            S, A, R, S_ = self.storage[i]
            s.append(np.array(S, copy=False))
            a.append(np.array(A, copy=False))

            r.append(np.array(R, copy=False))

            s_.append(np.array(S_, copy=False))

        return np.array(s), np.array(a), np.array(r), np.array(s_)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.storage, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.storage = pickle.load(f)
    
class Actor(nn.Module):
    def __init__(self, filter_num, padding_size):
        super(Actor, self).__init__()

        self.nx = padding_size[0]
        self.nz = padding_size[1]
        in_channels = filter_num[0]
        out_channels = filter_num[1]
        kernel_size = (self.nx * 2 + 1, self.nz * 2 + 1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0),
        )

    def periodic_pad(self, X, nx, nz):
        if nx != 0:
            X = torch.cat([X[:,:,-nx:,:], X, X[:,:,0:nx,:]], dim=2)
        if nz != 0:
            X = torch.cat([X[:,:,:,-nz:], X, X[:,:,:,0:nz]], dim=3)
        return X

    def forward(self, x):
        x = self.periodic_pad(x, self.nx, self.nz)
        x = self.layer1(x)
        return x
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )
        self.layer6 = None

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        if self.layer6 == None:
            num_input_features = x.size(1)
            self.layer6 = nn.Linear(num_input_features, 1)
        x = self.layer6(x)
        return x

###############################  DDPG  ####################################

class myDDPG(object):
    def __init__(self):
        # hyper parameters
        self.MAX_EP_STEPS = 1200
        self.state_step = 1
        self.LR_A = 0.001    # learning rate for actor
        self.LR_C = 0.002    # learning rate for critic
        self.GAMMA = 0.9     # reward discount
        self.TAU = 0.01      # soft replacement
        self.alpha = 0.01     # sigma of space noise
        self.MEMORY_CAPACITY = 20000
        self.learn_frequency = 5
        self.BATCH_SIZE = 64
        self.filter_num = [1, 1]
        self.padding_size = [0, 9]
        self.pointer = 0
        
        self.actor = Actor(self.filter_num, self.padding_size)
        self.actor_target = Actor(self.filter_num, self.padding_size)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_A)

        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_C)

        self.replay_buffer = Replay_buffer(self.MEMORY_CAPACITY)

    def choose_action(self, state):
        state = torch.FloatTensor(state[np.newaxis, np.newaxis, :, :])
        action = self.actor(state)
        return action.detach().numpy().squeeze()

    def learn(self):
        # Sample replay buffer
        s, a, r, s_ = self.replay_buffer.sample(self.BATCH_SIZE)
        state = torch.FloatTensor(s[:, np.newaxis, :, :])
        action = torch.FloatTensor(a[:, np.newaxis, :, :])
        reward = torch.FloatTensor(r[:, np.newaxis])
        next_state = torch.FloatTensor(s_[:, np.newaxis, :, :])

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (self.GAMMA * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.pointer % self.learn_frequency == 0:
            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            print('----------Actor Updated----------')
            weights = self.actor.state_dict()['layer1.0.weight'].numpy()
            np.savetxt('weights.plt', np.squeeze(weights))

        # Update the frozen target models
        # soft target replacement
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def save(self,num_epoch):
        torch.save(self.actor.state_dict(), './model/actor' + str(num_epoch) + '.pth')
        torch.save(self.critic.state_dict(), './model/critic' + str(num_epoch) + '.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self,num_epoch):
        self.actor.load_state_dict(torch.load('./model/actor' + str(num_epoch) + '.pth'))
        self.critic.load_state_dict(torch.load('./model/critic' + str(num_epoch) + '.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")