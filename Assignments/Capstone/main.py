from ddpg_agent import DDPGAgent
import argparse

import gym

ENV = 'Pendulum-v0'
MODEL_PATH = './saved_models/ddpg-best-model.h5'
TRAIN_EPISODES = 200
TEST_EPISODES = 10
MEMORY_SIZE = 50000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.002

env = gym.make(ENV)

def train():
    model = DDPGAgent(env = env, train_flag = True, num_episodes = TRAIN_EPISODES,
                    model_path = MODEL_PATH, actor_learning_rate = ACTOR_LEARNING_RATE,
                    critic_learning_rate = CRITIC_LEARNING_RATE, gamma = GAMMA,
                    memory_size = MEMORY_SIZE, batch_size = BATCH_SIZE, tau=TAU)
    model.train(render = False)


def test():
    model = DDPGAgent(env = env, train_flag = False, num_episodes = TEST_EPISODES,
                    model_path = MODEL_PATH)

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
