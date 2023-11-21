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


class DQN():
    def __init__(self, envName, seed, gamma,
                 bufferSize,
                 batchSize,
                 optimizerFn,
                 optimizerLR,
                 maxTrainEpisodes, maxEvalEpisodes,
                 explorationStrategyTrain,
                 explorationStrategyEval,
                 updateFrequency):

        self.gamma = gamma
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.updateFrequency = updateFrequency
        self.maxTrainEpisodes = maxTrainEpisodes
        self.maxEvalEpisodes = maxEvalEpisodes

        self.env = gym.make(envName)
        self.env.seed(seed)

        self.initBookKeeping()
        self.policyNetwork = createValueNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n, [128], F.relu).to(device)
        self.targetNetwork = createValueNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n, [128], F.relu).to(device)
        self.updateNetwork()
        self.targetNetwork.eval()

        self.optimizer = optimizerFn(
            self.policyNetwork.parameters(), optimizerLR)

        self.explorationStrategyTrain = explorationStrategyTrain
        self.explorationStrategyEval = explorationStrategyEval

        self.replayBuffer = ReplayBuffer(bufferSize)

    #################################################################################

    def initBookKeeping(self):
        # This method creates and initializes all the variables required for book-keeping values and it is called 'init' method

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

    #################################################################################

    # trainReward, trainTime, evalReward, evalSteps, wallClockTimeStart,
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

    #################################################################################

    def runDQN(self):
        self.trainAgent()
        self.finalEvalReward, _ = self.evaluateAgent()

        return self.trainRewardList, self.trainTimeList, self.evalRewardList, self.wallClockTimeList, self.evalStepList, self.finalEvalReward
        # return trainRewardsList, trainTimeList, evalRewardsList, wallClockTimeList, finalEvalReward

    #################################################################################

    def trainAgent(self):
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

    #################################################################################

    def trainNetwork(self, experiences, epochs):
        statesTensor, actionsTensor, rewardsTensor, nextStatesTensor, donesTensor = self.replayBuffer.splitExperiences(
            experiences)

        for _ in range(epochs):
            max_q_nextStates = self.targetNetwork(
                nextStatesTensor).max(1)[0].detach()

            # TODO: Toggle with (1-donesTensor)
            tdTargets = rewardsTensor + self.gamma * \
                max_q_nextStates * (1-donesTensor)

            tdTargets = tdTargets.detach()

            q_states = self.policyNetwork(
                statesTensor).gather(1, actionsTensor).squeeze(1)

            # tdErrors = tdTargets - q_states
            # loss = torch.mean(0.5*(tdErrors**2))
            loss = F.smooth_l1_loss(q_states, tdTargets)

            self.optimizer.zero_grad()
            loss.backward()
            # for param in self.policyNetwork.parameters():
            #     param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    #################################################################################

    def updateNetwork(self):
        # This function updates the onlineNetwork with the target network
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())

    #################################################################################

    def evaluateAgent(self):
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


envName = envNames[1]
seed = 1

dqnAgent = DQN(envName=envName, seed=seed, gamma=GAMMA_DQN[envName],
               bufferSize=BUFFER_SIZE_DQN[envName],
               batchSize=BATCH_SIZE_DQN[envName],
               optimizerFn=optim.Adam,
               optimizerLR=LR_DQN[envName],
               maxTrainEpisodes=MAX_TRAIN_EPISODES_DQN[envName], maxEvalEpisodes=MAX_EVAL_EPISODES_DQN[envName],
               explorationStrategyTrain=EpsilonGreedyExponential(
                   EPSILON_START, 0.01, 256),
               explorationStrategyEval=EpsilonGreedyExponential(
                   EPSILON_START, 0.01, 256),
               updateFrequency=UPDATE_FREQUENCY_DQN[envName])

trainRewardList, trainTimeList, evalRewardList, wallClockTimeList, evalStepList, finalEvalReward = dqnAgent.runDQN()

plotQuantity([np.array([trainRewardList])], MAX_TRAIN_EPISODES, {
             'title': 'Training Reward vs Episodes', 'xLabel': 'Episodes', 'yLabel': 'Training Reward', 'legend': ['DQN']})
plotQuantity([np.array([evalRewardList])], MAX_TRAIN_EPISODES, {
             'title': 'Evaluation Reward vs Episodes', 'xLabel': 'Episodes', 'yLabel': 'Evaluation Reward', 'legend': ['DQN']})

# plotQuantity([np.array([trainTimeList])], MAX_TRAIN_EPISODES, {
#              'title': 'Training Reward vs Episodes', 'xLabel': 'Episodes', 'yLabel': 'Training Time', 'legend': ['DQN']})
# plotQuantity([np.array([wallClockTimeList])], MAX_TRAIN_EPISODES, {
#              'title': 'Training Reward vs Episodes', 'xLabel': 'Episodes', 'yLabel': 'Wall Clock Time', 'legend': ['DQN']})
# plotQuantity([np.array([evalStepList])], MAX_TRAIN_EPISODES, {
#              'title': 'Training Reward vs Episodes', 'xLabel': 'Episodes', 'yLabel': 'Evaluation Step', 'legend': ['DQN']})
