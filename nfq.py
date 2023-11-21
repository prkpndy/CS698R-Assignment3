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

from valueNetwork import createValueNetwork

#################################################################################

from constants import *

#################################################################################

from plotResults import plotQuantity

#################################################################################

from decay import EpsilonGreedyExponential

#################################################################################


class NFQ():
    def __init__(self, envName, seed, gamma, epochs,
                 bufferSize,
                 batchSize,
                 optimizerFn,
                 optimizerLR,
                 maxTrainEpisodes, maxEvalEpisodes,
                 explorationStrategyTrain,
                 explorationStrategyEval):
        # This NFQ method
        # 1. creates and initializes (with seed) the environment, train/eval episodes, gamma, etc.
        # 2. creates and initializes all the variables required for book-keeping values via the initBookKeeping method
        # 3. creates Q-network using the createValueNetwork above
        # 4. creates and initializes (with network params) the optimizer function
        # 5. sets the explorationStrategy variables/functions for train and evaluation
        # 6. sets the batchSize for the number of experiences
        # 7. Creates the replayBuffer

        self.gamma = gamma
        self.epochs = epochs
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.maxTrainEpisodes = maxTrainEpisodes
        self.maxEvalEpisodes = maxEvalEpisodes

        self.env = gym.make(envName)
        self.env.seed(seed)

        self.initBookKeeping()
        self.policyNetwork = createValueNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n, [16, 16], F.relu).to(device)

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

    def runNFQ(self):
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

    def trainAgent(self):
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

            # NOTE: batchSize and bufferSize are same in this case. Hence, essentially their is no sampling
            experiences = self.replayBuffer.sample(self.batchSize)
            self.trainNetwork(experiences, self.epochs)

            trainTimeEnd = time.time()

            self.performBookKeeping(
                train=True, trainReward=trainReward, trainTime=trainTimeEnd - trainTimeStart)

            evalReward, evalSteps = self.evaluateAgent()

            self.performBookKeeping(
                train=False, evalReward=evalReward, evalSteps=evalSteps, wallClockTimeStart=trainTimeStart)

        print(
            f'====> episode = {e}, epsilon = {self.explorationStrategyTrain.next()}, length = {self.replayBuffer.length()}, Performance = {self.evaluateAgent()}')

    def trainNetwork(self, experiences, epochs):
        statesTensor, actionsTensor, rewardsTensor, nextStatesTensor, donesTensor = self.replayBuffer.splitExperiences(
            experiences)

        for _ in range(epochs):
            # max_q_nextStates = self.policyNetwork(
            #     nextStatesTensor).max(1)[0].detach()
            max_q_nextStates = self.policyNetwork(
                nextStatesTensor).max(1)[0]

            # TODO: Toggle with (1-donesTensor)
            tdTargets = rewardsTensor + self.gamma * max_q_nextStates

            # tdTargets = tdTargets.detach()

            q_states = self.policyNetwork(
                statesTensor).gather(1, actionsTensor).squeeze(1)

            loss = F.smooth_l1_loss(q_states, tdTargets)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

    def evaluateAgent(self):
        rewards = []
        numStepsList = []

        for e in range(MAX_EVAL_EPISODES):
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

nfqAgent = NFQ(envName=envName, seed=seed, gamma=GAMMA_NFQ[envName], epochs=EPOCH_NFQ[envName],
               bufferSize=BUFFER_SIZE_NFQ[envName],
               batchSize=BATCH_SIZE_NFQ[envName],
               optimizerFn=optim.Adam,
               optimizerLR=LR_NFQ[envName],
               maxTrainEpisodes=MAX_TRAIN_EPISODES_NFQ[envName], maxEvalEpisodes=MAX_EVAL_EPISODES_NFQ[envName],
               explorationStrategyTrain=EpsilonGreedyExponential(
                   EPSILON_START, 0.01, 150),
               explorationStrategyEval=EpsilonGreedyExponential(EPSILON_START, 0.01, 150))

trainRewardList, trainTimeList, evalRewardList, wallClockTimeList, evalStepList, finalEvalReward = nfqAgent.runNFQ()

plotQuantity([np.array([trainRewardList])], MAX_TRAIN_EPISODES, {
             'title': 'Training Reward vs Episodes', 'xLabel': 'Episodes', 'yLabel': 'Training Reward', 'legend': ['DQN']})
# plotQuantity([np.array([trainTimeList])], MAX_TRAIN_EPISODES, {
#              'title': 'Training Reward vs Episodes', 'xLabel': 'Episodes', 'yLabel': 'Training Time', 'legend': ['DQN']})
plotQuantity([np.array([evalRewardList])], MAX_TRAIN_EPISODES, {
             'title': 'Training Reward vs Episodes', 'xLabel': 'Episodes', 'yLabel': 'Evaluation Reward', 'legend': ['DQN']})
# plotQuantity([np.array([wallClockTimeList])], MAX_TRAIN_EPISODES, {
#              'title': 'Training Reward vs Episodes', 'xLabel': 'Episodes', 'yLabel': 'Wall Clock Time', 'legend': ['DQN']})
# plotQuantity([np.array([evalStepList])], MAX_TRAIN_EPISODES, {
#              'title': 'Training Reward vs Episodes', 'xLabel': 'Episodes', 'yLabel': 'Evaluation Step', 'legend': ['DQN']})
