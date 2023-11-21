import torch
import numpy as np

import random
from collections import deque
import copy

from constants import device

# TODO: Implement betaRate


class ReplayBuffer():
    def __init__(self, bufferSize, bufferType='DQN', **kwargs):
        self.bufferType = bufferType
        self.bufferSize = bufferSize
        self.memory = deque([], maxlen=self.bufferSize)

        if self.bufferType == 'PER-D3QN':
            self.priorities = deque([], maxlen=self.bufferSize)
            self.alpha = kwargs['alpha']
            self.beta = kwargs['beta']
            self.betaRate = kwargs['betaRate']
            self.epsilon = kwargs['epsilon']

    #################################################################################

    def store(self, experience):
        self.memory.append(experience)

        if self.bufferType == 'PER-D3QN':
            self.priorities.append(max(self.priorities, default=1))

    #################################################################################

    def update(self, indices, priorities):
        # This is mainly used for Prioritized Experience Replay
        # Otherwise just have a pass in this method
        #
        # This function does not return anything

        for i, v in enumerate(indices):
            self.priorities[v] = priorities[i] + self.epsilon

    #################################################################################

    def collectExperiences(self, env, state, explorationStrategy, countExperiences, net=None):
        s = np.copy(state)

        totalReward = 0
        for numSteps in range(countExperiences):
            epsilon = explorationStrategy.next()

            a = np.random.randint(0, env.action_space.n)
            if np.random.random() > epsilon:
                t = torch.tensor(np.copy(s), device=device,
                                 dtype=torch.float32)
                with torch.no_grad():
                    a = int(net(t).max(0)[1].view(1)[0])

            nextState, reward, done, _ = env.step(a)
            totalReward += reward

            # if done:
            #     if numSteps < 30:
            #         self.store([s, a, reward-10, nextState, done])
            #     else:
            #         self.store([s, a, reward, nextState, done])

            #     break

            self.store([s, a, reward, nextState, done])

            s = nextState

            if done:
                break

        return totalReward

    #################################################################################

    def sample(self, batchSize, **kwargs):
        sampleSize = min(self.length(), batchSize)

        if self.bufferType == 'PER-D3QN':
            scaledPriorities = np.power(np.array(self.priorities), self.alpha)
            probabilities = scaledPriorities/np.sum(scaledPriorities)

            sampleIndices = np.random.choice(
                range(self.length()), size=sampleSize, replace=False, p=probabilities)
            experiencesList = [self.memory[i] for i in sampleIndices]
            sampleProbabilities = probabilities[sampleIndices]

            weights = np.power((1/self.length()) *
                               (1/sampleProbabilities), self.beta)
            weights = weights/np.max(weights)

            self.beta *= self.betaRate

            return experiencesList, weights, sampleIndices
        else:
            experiencesList = random.sample(self.memory, batchSize)

            return experiencesList

        # if self.length() < batchSize:
        #     experiencesList = list(self.memory)
        # else:
        #     experiencesList = random.sample(self.memory, batchSize)

        # return experiencesList

    #################################################################################

    def splitExperiences(self, experiences):
        numRows = len(experiences)

        states = torch.zeros(
            (numRows, experiences[0][0].size), device=device, dtype=torch.float32)
        actions = torch.zeros((numRows, 1), device=device, dtype=torch.int64)
        rewards = torch.zeros(numRows, device=device, dtype=torch.float32)
        nextStates = torch.zeros(
            (numRows, experiences[0][3].size), device=device, dtype=torch.float32)
        dones = torch.zeros(numRows, device=device, dtype=torch.int64)

        for i in range(numRows):
            states[i] = torch.tensor(
                experiences[i][0], device=device, dtype=torch.float32)
            actions[i] = torch.tensor(
                [experiences[i][1]], device=device, dtype=torch.int64)
            rewards[i] = torch.tensor(
                experiences[i][2], device=device, dtype=torch.float32)
            nextStates[i] = torch.tensor(
                experiences[i][3], device=device, dtype=torch.float32)
            dones[i] = torch.tensor(
                experiences[i][4], device=device, dtype=torch.int64)

        return states.detach(), actions.detach(), rewards.detach(), nextStates.detach(), dones.detach()

    #################################################################################

    def length(self):
        return len(self.memory)
