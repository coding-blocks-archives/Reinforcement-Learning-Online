from a2c_agent import A2CAgent
from pong import PongEnv
import argparse

ENV = 'PongDeterministic-v4'
MODEL_PATH = './saved_models/a2c-best_model.model'
TRAIN_EPISODES = 1000
TEST_EPISODES = 10
GAMMA = 0.99
ACTOR_LEARNING_RATE = 0.00025
CRITIC_LEARNING_RATE = 0.00025
NUM_CHECKPOINTS = 10

env = PongEnv(ENV)

def train():
    model = A2CAgent(env = env, train_flag = True, num_episodes = TRAIN_EPISODES,
                    model_path = MODEL_PATH, actor_learning_rate = ACTOR_LEARNING_RATE,
                    critic_learning_rate = CRITIC_LEARNING_RATE, gamma = GAMMA, 
                    num_checkpoints = NUM_CHECKPOINTS)
    model.train(render = False)


def test():
    model = A2CAgent(env = env, train_flag = False, num_episodes = TEST_EPISODES,
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
