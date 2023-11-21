# Mention the values of all the hyperparameters to be used in the entire notebook, put the values that gave the best
# performance and were finally used for the agent

device = 'cpu'
envNames = ['CartPole-v0', 'MountainCar-v0']

MAX_TRAIN_EPISODES = 1000
MAX_EVAL_EPISODES = 2
EPSILON_START = 0.99

GAMMA_NFQ = {'CartPole-v0': 0.95, 'MountainCar-v0': 0.95}
EPOCH_NFQ = {'CartPole-v0': 2, 'MountainCar-v0': 2}
BUFFER_SIZE_NFQ = {'CartPole-v0': 512, 'MountainCar-v0': 512}
BATCH_SIZE_NFQ = {'CartPole-v0': 512, 'MountainCar-v0': 512}
LR_NFQ = {'CartPole-v0': 0.0001, 'MountainCar-v0': 0.0001}
MAX_TRAIN_EPISODES_NFQ = {'CartPole-v0': 5000, 'MountainCar-v0': 5000}
MAX_EVAL_EPISODES_NFQ = {
    'CartPole-v0': MAX_EVAL_EPISODES, 'MountainCar-v0': MAX_EVAL_EPISODES}

GAMMA_DQN = {'CartPole-v0': 0.95, 'MountainCar-v0': 0.95}
EPOCH_DQN = {'CartPole-v0': 2, 'MountainCar-v0': 2}
BUFFER_SIZE_DQN = {'CartPole-v0': 5000, 'MountainCar-v0': 5000}
BATCH_SIZE_DQN = {'CartPole-v0': 512, 'MountainCar-v0': 512}
LR_DQN = {'CartPole-v0': 0.001, 'MountainCar-v0': 0.001}
MAX_TRAIN_EPISODES_DQN = {
    'CartPole-v0': MAX_TRAIN_EPISODES, 'MountainCar-v0': MAX_TRAIN_EPISODES}
MAX_EVAL_EPISODES_DQN = {
    'CartPole-v0': MAX_EVAL_EPISODES, 'MountainCar-v0': MAX_EVAL_EPISODES}
UPDATE_FREQUENCY_DQN = {'CartPole-v0': 50, 'MountainCar-v0': 50}

GAMMA_DDQN = {'CartPole-v0': 0.95, 'MountainCar-v0': 0.95}
EPOCH_DDQN = {'CartPole-v0': 2, 'MountainCar-v0': 2}
BUFFER_SIZE_DDQN = {'CartPole-v0': 5000, 'MountainCar-v0': 5000}
BATCH_SIZE_DDQN = {'CartPole-v0': 512, 'MountainCar-v0': 512}
LR_DDQN = {'CartPole-v0': 0.001, 'MountainCar-v0': 0.001}
MAX_TRAIN_EPISODES_DDQN = {
    'CartPole-v0': MAX_TRAIN_EPISODES, 'MountainCar-v0': MAX_TRAIN_EPISODES}
MAX_EVAL_EPISODES_DDQN = {
    'CartPole-v0': MAX_EVAL_EPISODES, 'MountainCar-v0': MAX_EVAL_EPISODES}
UPDATE_FREQUENCY_DDQN = {'CartPole-v0': 80, 'MountainCar-v0': 80}

GAMMA_D3QN = {'CartPole-v0': 0.95, 'MountainCar-v0': 0.95}
TAU_D3QN = {'CartPole-v0': 0.5, 'MountainCar-v0': 0.5}
EPOCH_D3QN = {'CartPole-v0': 2, 'MountainCar-v0': 2}
BUFFER_SIZE_D3QN = {'CartPole-v0': 5000, 'MountainCar-v0': 5000}
BATCH_SIZE_D3QN = {'CartPole-v0': 512, 'MountainCar-v0': 512}
LR_D3QN = {'CartPole-v0': 0.001, 'MountainCar-v0': 0.001}
MAX_TRAIN_EPISODES_D3QN = {
    'CartPole-v0': MAX_TRAIN_EPISODES, 'MountainCar-v0': MAX_TRAIN_EPISODES}
MAX_EVAL_EPISODES_D3QN = {
    'CartPole-v0': MAX_EVAL_EPISODES, 'MountainCar-v0': MAX_EVAL_EPISODES}
UPDATE_FREQUENCY_D3QN = {'CartPole-v0': 80, 'MountainCar-v0': 80}

GAMMA_PERD3QN = {'CartPole-v0': 0.95, 'MountainCar-v0': 0.95}
TAU_PERD3QN = {'CartPole-v0': 0.5, 'MountainCar-v0': 0.5}
ALPHA_PERD3QN = {'CartPole-v0': 0.6, 'MountainCar-v0': 0.6}
BETA_PERD3QN = {'CartPole-v0': 0.1, 'MountainCar-v0': 0.1}
BETA_RATE_PERD3QN = {'CartPole-v0': 0.9995, 'MountainCar-v0': 0.9995}
EPSILON_PERD3QN = {'CartPole-v0': 0.01, 'MountainCar-v0': 0.01}
EPOCH_PERD3QN = {'CartPole-v0': 2, 'MountainCar-v0': 2}
BUFFER_SIZE_PERD3QN = {'CartPole-v0': 4000, 'MountainCar-v0': 4000}
BATCH_SIZE_PERD3QN = {'CartPole-v0': 512, 'MountainCar-v0': 512}
LR_PERD3QN = {'CartPole-v0': 0.0007, 'MountainCar-v0': 0.0007}
MAX_TRAIN_EPISODES_PERD3QN = {
    'CartPole-v0': MAX_TRAIN_EPISODES, 'MountainCar-v0': MAX_TRAIN_EPISODES}
MAX_EVAL_EPISODES_PERD3QN = {
    'CartPole-v0': MAX_EVAL_EPISODES, 'MountainCar-v0': MAX_EVAL_EPISODES}
UPDATE_FREQUENCY_PERD3QN = {'CartPole-v0': 30, 'MountainCar-v0': 30}
