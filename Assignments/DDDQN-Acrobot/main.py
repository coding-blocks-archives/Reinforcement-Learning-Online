import gym
import argparse
from dddqn_agent import DuelingDoubleDeepQNetworkAgent

ENV = 'Acrobot-v1'
MODEL_PATH = './saved_models/dddqn-final.model'
BEST_MODEL_PATH = './saved_models/dddqn-best_model.model'
TRAIN_EPISODES = 1000
TEST_EPISODES = 10
MEMORY_SIZE = 10000
NUM_CHECKPOINTS = 10
BATCH_SIZE = 32
GAMMA = 0.95
EPS_START = 1.0
EPS_MIN = 0.001
LEARNING_RATE = 0.00025
EXPRORATION_PHASE = 0.3
TRAIN_START_EPISODE = 20
TARGET_MODEL_UPDATE_INTERVAL = 5

env = gym.make(ENV)

def train():
    model = DuelingDoubleDeepQNetworkAgent(env = env, train_flag = True, model_path = MODEL_PATH, memory_size = MEMORY_SIZE,\
                            num_checkpoints = NUM_CHECKPOINTS, num_episodes = TRAIN_EPISODES, batch_size = BATCH_SIZE, \
                            train_start_episode = TRAIN_START_EPISODE, exploration_phase = EXPRORATION_PHASE,  \
                            learning_rate = LEARNING_RATE, gamma = GAMMA, eps_start = EPS_START, eps_min = EPS_MIN, \
                            target_model_update_interval = TARGET_MODEL_UPDATE_INTERVAL)
    model.train(render = False)


def test():
    model = DuelingDoubleDeepQNetworkAgent(env, train_flag = False, model_path = BEST_MODEL_PATH, num_episodes = TEST_EPISODES)

    model.test(render = True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices = ['train', 'test'], default = 'train', help = 'Train  or test mode')
    

    args = parser.parse_args()

    if(args.mode == 'test'):
        test()

    elif(args.mode == 'train'):
        train()



if __name__ == "__main__":
    main()