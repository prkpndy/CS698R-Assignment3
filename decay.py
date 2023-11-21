import numpy as np


class EpsilonGreedyExponential:
    def __init__(self, initialValue, finalValue, numSteps):
        self.initialValue = initialValue
        self.finalValue = finalValue
        self.numSteps = numSteps
        self.rate = np.power((initialValue/finalValue), (1/(numSteps-1)))
        self.num = 0

    def reset(self):
        self.num = 0

    def next(self):
        if self.num == self.numSteps:
            return self.finalValue
        value = self.initialValue/np.power(self.rate, self.num)
        self.num += 1
        return value
