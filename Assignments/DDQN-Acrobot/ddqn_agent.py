from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.losses import Huber
from replay_memory import Memory
import numpy as np

def get_model(input_shape, num_actions, learning_rate):
    X_inp = Input(shape = (input_shape,))
    X = Dense(32, activation='relu', kernel_initializer='he_uniform')(X_inp)
    X = Dense(32, activation='relu', kernel_initializer='he_uniform')(X)
    X = Dense(num_actions, activation='linear', kernel_initializer='he_uniform')(X)
    model = Model(inputs = X_inp, outputs = X, name = 'DDQN-model')
    print(model.summary())
    opt = RMSprop(learning_rate = learning_rate)
    model.compile(optimizer = opt, loss = Huber(delta = 1.5))
    return model

    

class DoubleDeepQNetworkAgent(object):
    def __init__(self, env, \
                train_flag = True, \
                model_path = None, \
                memory_size = 512, \
                num_checkpoints = 5, \
                num_episodes = 1000, \
                batch_size = 64, \
                learning_rate = 0.01, \
                gamma = 1.0, \
                exploration_phase = 0.4, \
                train_start_episode = 100, \
                eps_start = 1.0, \
                eps_min = 0.05, \
                eps_decay = 0.999, \
                target_model_update_interval = 20):
        self.env = env
        self.train_flag = train_flag
        self.model_path = model_path
        self.input_shape = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n 
        self.num_episodes = num_episodes
        self.num_checkpoints = num_checkpoints
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.gamma = gamma
        self.train_start_episode = train_start_episode
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = (eps_start - eps_min)/((num_episodes - train_start_episode) * exploration_phase)
        self.target_model_update_interval = target_model_update_interval

        if(self.train_flag):
            self.model = get_model(input_shape = self.input_shape, num_actions = self.num_actions, learning_rate = self.learning_rate)
            self.target_model = get_model(input_shape = self.input_shape, num_actions = self.num_actions, learning_rate = self.learning_rate)
            self.memory = Memory(self.memory_size)
        else:
            assert model_path != None, "Please pass the path of a trained model!"
            self.model = load_model(model_path)
            print('Model Loaded!!')

    def get_reward(self, state, done, t):
        if(done and t+1 < self.env._max_episode_steps):
            return 1000
        if(state[0] < 0.2):
            return (2  - state[0])**2
        return 0

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        return


    def train(self, render = False):
        self.all_episode_rewards = []
        self.all_epiosde_timestep = []
        best_score = float('inf')
        for episode in range(self.num_episodes):
            loss = 0
            reward = 0
            episode_reward = 0
            state = self.env.reset()
            state = np.reshape(state, [1, self.input_shape])
            t = 0
            while(1):
                if(render):
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                reward = self.get_reward(next_state, done, t)
                # reward = self.get_reward(next_state)
                next_state = np.reshape(next_state, [1, self.input_shape])
                episode_reward = episode_reward + reward
                self.memory.add_record(state, action, reward, next_state, done)

                if(episode >= self.train_start_episode):
                    batch, batch_size = self.memory.get_batch(self.batch_size)
                    state_batch = np.zeros((batch_size, self.input_shape))
                    action_batch = np.zeros(batch_size, dtype = int)
                    reward_batch = np.zeros(batch_size)
                    next_state_batch = np.zeros((batch_size, self.input_shape))
                    done_batch = np.zeros(batch_size)


                    for i, record in enumerate(batch):
                        state_batch[i] = record['state']
                        action_batch[i] = int(record['action'])
                        reward_batch[i] = record['reward']
                        next_state_batch[i] = record['next_state']
                        done_batch[i] = record['done']
                    
                    targets = self.model.predict(state_batch)
                    targets_next_state = self.model.predict(next_state_batch)
                    target_model_next_state = self.target_model.predict(next_state_batch)

                    for i in range(batch_size):
                        if(done_batch[i]):
                            targets[i][action_batch[i]] = reward_batch[i]
                        else:
                            a = np.argmax(targets_next_state[i])
                            targets[i][action_batch[i]] = reward_batch[i] + self.gamma * target_model_next_state[i][a]
            
                    self.model.train_on_batch(state_batch, targets)
                    
                
                state = next_state

                t = t + 1

                if(done or t>=1000):
                    self.all_episode_rewards.append(episode_reward)
                    self.all_epiosde_timestep.append(t)
                    print("Episode {}/{} | Episode steps : {} ({:.4}) | Episode reward : {} ({:.4}) | Epsilon : {:.4}".format(episode+1, self.num_episodes, t, np.mean(self.all_epiosde_timestep[-10:]),  episode_reward, np.mean(self.all_episode_rewards[-10:]), self.eps))
                    if(np.mean(self.all_epiosde_timestep[-10:]) < best_score):
                        best_score = np.mean(self.all_epiosde_timestep[-10:])
                        self.model.save('./saved_models/ddqn-best_model.model')
                        print('Best Model Saved !')
                    break
            
            if(episode > self.train_start_episode and episode%self.target_model_update_interval == 0):
                self.update_target_model()

            self.env.reset()
                

            if(self.num_checkpoints != 0 and (episode % (self.num_episodes/self.num_checkpoints)) == 0):
                self.model.save('./saved_models/ddqn-{:06d}.model'.format(episode))
            
            if(episode >= self.train_start_episode):
                self.eps = max(self.eps - self.eps_decay, self.eps_min)
        
        
        if(self.model_path != None):
            print('Saving model at this path : {}'.format(self.model_path))
            self.model.save(self.model_path)
        else:
            print('Saving model at this path : ./saved_models/ddqn-final.model')
            self.model.save('./saved_models/ddqn-final.model')
    
    def test(self, render = True):
        for episode in range(self.num_episodes):
            loss = 0
            reward = 0
            episode_reward = 0
            state = self.env.reset()
            while(1):
                if(render):
                    self.env.render()
                state = np.reshape(state, [1, self.input_shape])
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward = episode_reward + reward
                state = next_state

                if(done):
                    print("Episode {}/{} | Episode reward : {}".format(episode+1, self.num_episodes, episode_reward))
                    break
            self.env.close()



    def get_action(self, state):
        best_action = np.argmax(self.model.predict(state))
        if(self.train_flag):
            random_action = np.random.randint(self.num_actions)
            eps_i = np.random.random()
            if(eps_i < self.eps):
                return random_action
            else:
                return best_action
        else:
            return best_action



















