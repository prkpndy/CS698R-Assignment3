import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
from collections import deque
import copy
import time

#################################################################################

from replayBuffer import ReplayBuffer

#################################################################################

from duelingNetwork import createDuelingNetwork

#################################################################################

from constants import *

#################################################################################

from plotResults import plotQuantity

#################################################################################

from decay import EpsilonGreedyExponential

#################################################################################


class D3QN():
    def __init__(self, envName, seed, gamma, tau,
                 bufferSize,
                 batchSize,
                 optimizerFn,
                 optimizerLR,
                 maxTrainEpisodes, maxEvalEpisodes,
                 explorationStrategyTrain,
                 explorationStrategyEval,
                 updateFrequency):
        # This D3QN method
        # 1. creates and initializes (with seed) the environment, train/eval episodes, gamma, etc.
        # 2. creates and initializes all the variables required for book-keeping values via the initBookKeeping method
        # 3. creates tareget and online Q-networks using the createValueNetwork above
        # 4. creates and initializes (with network params) the optimizer function
        # 5. sets the explorationStrategy variables/functions for train and evaluation
        # 6. sets the batchSize for the number of experiences
        # 7. Creates the replayBuffer

        self.gamma = gamma
        self.tau = tau
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.updateFrequency = updateFrequency
        self.maxTrainEpisodes = maxTrainEpisodes
        self.maxEvalEpisodes = maxEvalEpisodes

        self.env = gym.make(envName)
        self.env.seed(seed)

        self.initBookKeeping()

        self.policyNetwork = createDuelingNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n, {'valueStream': [16, 16], 'advantageStream': [24, 16]}, F.relu).to(device)
        self.targetNetwork = createDuelingNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n, {'valueStream': [16, 16], 'advantageStream': [24, 16]}, F.relu).to(device)
        self.updateNetwork()
        self.targetNetwork.eval()

        self.optimizer = optimizerFn(
            self.policyNetwork.parameters(), optimizerLR)

        self.explorationStrategyTrain = explorationStrategyTrain
        self.explorationStrategyEval = explorationStrategyEval

        self.replayBuffer = ReplayBuffer(bufferSize)

    def initBookKeeping(self):
        # This method creates and initializes all the variables required for book-keeping values and it is called init method

        self.evalStepList = np.zeros(self.maxTrainEpisodes)
        self.trainRewardList = np.zeros(self.maxTrainEpisodes)
        self.evalRewardList = np.zeros(self.maxTrainEpisodes)
        self.trainTimeList = np.zeros(
            self.maxTrainEpisodes)  # Time for training
        # Time for training + evaluation + bookkeeping
        self.wallClockTimeList = np.zeros(self.maxTrainEpisodes)

        self.finalEvalReward = 0

        self.lastTrainTime = 0
        self.lastWallClockTime = 0

        self.numEpisode = -1

    def performBookKeeping(self, train=True, **kwargs):
        # This method updates relevant variables for the bookKeeping, this can be called multiple times during training
        # If you want you can print information using this, so it may help to monitor progress and also help to debug

        if train:
            self.trainRewardList[self.numEpisode] = kwargs['trainReward']
            self.trainTimeList[self.numEpisode] = kwargs['trainTime'] + \
                self.lastTrainTime
            self.lastTrainTime += kwargs['trainTime']
        else:
            self.evalRewardList[self.numEpisode] = kwargs['evalReward']
            self.evalStepList[self.numEpisode] = kwargs['evalSteps']
            wallClockTimeEnd = time.time()
            wallClockTime = wallClockTimeEnd - kwargs['wallClockTimeStart']
            self.wallClockTimeList[self.numEpisode] = wallClockTime + \
                self.lastWallClockTime
            self.lastWallClockTime += wallClockTime

    def runD3QN(self):
        # This is the main method, it trains the agent, performs bookkeeping while training and finally evaluates
        # the agent and returns the following quantities:
        # 1. episode wise mean train rewards
        # 2. episode wise mean eval rewards
        # 2. episode wise trainTime (in seconds): time elapsed during training since the start of the first episode
        # 3. episode wise wallClockTime (in seconds): actual time elapsed since the start of training,
        # Note: This will include time for BookKeeping and evaluation
        # Note: both trainTime and wallClockTime get accumulated as episodes proceed.

        self.trainAgent()
        self.finalEvalReward, _ = self.evaluateAgent()

        return self.trainRewardList, self.trainTimeList, self.evalRewardList, self.wallClockTimeList, self.evalStepList, self.finalEvalReward

        # return trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, finalEvalReward

    def trainAgent(self):
        # This method collects experiences and trains the agent and does BookKeeping while training.
        # This calls the trainNetwork() method internally, it also evaluates the agent per episode
        # It trains the agent for MAX_TRAIN_EPISODES

        self.updateNetwork()

        for e in range(self.maxTrainEpisodes):
            self.numEpisode += 1

            if e % 100 == 0:
                print(
                    f'====> episode = {e}, epsilon = {self.explorationStrategyTrain.next()}, length = {self.replayBuffer.length()}, Performance = {self.evaluateAgent()}')

            # TODO: Maybe not required to reset the exploration strategy after every episode
            s = self.env.reset()
            self.explorationStrategyTrain.reset()

            trainTimeStart = time.time()

            trainReward = self.replayBuffer.collectExperiences(
                self.env, s, self.explorationStrategyTrain, self.bufferSize, self.policyNetwork)

            if self.replayBuffer.length() < self.batchSize:
                continue

            for _ in range(5):
                experiences = self.replayBuffer.sample(self.batchSize)
                self.trainNetwork(experiences, 1)

            trainTimeEnd = time.time()

            self.performBookKeeping(
                train=True, trainReward=trainReward, trainTime=trainTimeEnd - trainTimeStart)

            evalReward, evalSteps = self.evaluateAgent()

            self.performBookKeeping(
                train=False, evalReward=evalReward, evalSteps=evalSteps, wallClockTimeStart=trainTimeStart)

            if e % self.updateFrequency == 0:
                self.updateNetwork()

        print(
            f'====> episode = {e}, epsilon = {self.explorationStrategyTrain.next()}, length = {self.replayBuffer.length()}, Performance = {self.evaluateAgent()}')

        # return trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList

    def trainNetwork(self, experiences, epochs):
        # This method trains the value network epoch number of times and is called by the trainAgent function
        # It essentially uses the experiences to calculate target, using the targets it calculates the error, which
        # is further used for calculating the loss. It then uses the optimizer over the loss
        # to update the params of the network by backpropagating through the network
        # This function does not return anything
        # You can try out other loss functions other than MSE like Huber loss, MAE, etc.

        for _ in range(epochs):
            statesTensor, actionsTensor, rewardsTensor, nextStatesTensor, donesTensor = self.replayBuffer.splitExperiences(
                experiences)

            argmax_q_nextStates = self.policyNetwork(
                nextStatesTensor).max(1)[1].detach()

            q_nextStates = self.targetNetwork(nextStatesTensor).detach()

            max_q_nextStates = q_nextStates.gather(
                1, argmax_q_nextStates.unsqueeze(1)).squeeze(1).detach()

            # TODO: Toggle with (1-donesTensor)
            tdTargets = rewardsTensor + self.gamma * \
                max_q_nextStates * (1 - donesTensor)

            tdTargets = tdTargets.detach()

            q_states = self.policyNetwork(
                statesTensor).gather(1, actionsTensor).squeeze(1)

            loss = F.smooth_l1_loss(q_states, tdTargets)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

    def updateNetwork(self):
        # This function updates the onlineNetwork with the target network using Polyak averaging

        policyNetParams = self.policyNetwork.named_parameters()
        targetNetParams = self.targetNetwork.named_parameters()

        targetNetParamsNew = dict(targetNetParams)

        for name, value in policyNetParams:
            if name in targetNetParamsNew:
                targetNetParamsNew[name].data.copy_(
                    self.tau*value.data + (1 - self.tau)*targetNetParamsNew[name].data)

        self.targetNetwork.load_state_dict(targetNetParamsNew)

    def evaluateAgent(self):
        # This function evaluates the agent using the value network, it evaluates agent for MAX_EVAL_EPISODES
        # Typically MAX_EVAL_EPISODES = 1

        rewards = []
        numStepsList = []

        for e in range(self.maxEvalEpisodes):
            cumulativeReward = 0
            numSteps = 0

            s = self.env.reset()
            # self.explorationStrategyEval.reset()

            done = False
            while not done:
                # epsilon = self.explorationStrategyEval.next()
                epsilon = 0
                a = np.random.randint(0, self.env.action_space.n)
                if np.random.random() > epsilon:
                    t = torch.tensor(np.copy(s), device=device,
                                     dtype=torch.float32)
                    with torch.no_grad():
                        a = int(self.policyNetwork(t).max(0)[1].view(1)[0])

                nextState, reward, done, _ = self.env.step(a)
                s = nextState

                cumulativeReward += reward
                numSteps += 1

            rewards.append(cumulativeReward)
            numStepsList.append(numSteps)

        return np.mean(rewards), np.mean(numStepsList)


envName = envNames[0]
seed = 1

d3qnAgent = D3QN(envName=envName, seed=seed, gamma=GAMMA_D3QN[envName], tau=TAU_D3QN[envName],
                 bufferSize=BUFFER_SIZE_D3QN[envName],
                 batchSize=BATCH_SIZE_D3QN[envName],
                 optimizerFn=optim.Adam,
                 optimizerLR=LR_D3QN[envName],
                 maxTrainEpisodes=MAX_TRAIN_EPISODES_D3QN[envName], maxEvalEpisodes=MAX_EVAL_EPISODES_D3QN[envName],
                 explorationStrategyTrain=EpsilonGreedyExponential(
                     EPSILON_START, 0.01, 256),
                 explorationStrategyEval=EpsilonGreedyExponential(
                     EPSILON_START, 0.01, 256),
                 updateFrequency=UPDATE_FREQUENCY_D3QN[envName])

trainRewardList, trainTimeList, evalRewardList, wallClockTimeList, evalStepList, finalEvalReward = d3qnAgent.runD3QN()

plotQuantity([np.array([trainRewardList])], MAX_TRAIN_EPISODES, {
             'title': 'Training Reward vs Episodes', 'xLabel': 'Episodes', 'yLabel': 'Training Reward', 'legend': ['DQN']})
plotQuantity([np.array([evalRewardList])], MAX_TRAIN_EPISODES, {
             'title': 'Evaluation Reward vs Episodes', 'xLabel': 'Episodes', 'yLabel': 'Evaluation Reward', 'legend': ['DQN']})
