from keras.layers import Dense, Input, Conv2D, Flatten
from keras.models import Model, load_model
from keras.optimizers import RMSprop, Adam
from keras.losses import Huber
import numpy as np
import gym
from collections import deque
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as tfback
import keras.backend as K
import tensorflow as tf
import time

# def _get_available_gpus():
#     """Get a list of available gpu devices (formatted as strings).

#     # Returns
#         A list of available GPU devices.
#     """
#     #global _LOCAL_DEVICES
#     if tfback._LOCAL_DEVICES is None:
#         devices = tf.config.list_logical_devices()
#         tfback._LOCAL_DEVICES = [x.name for x in devices]
#     return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

# tfback._get_available_gpus = _get_available_gpus

def get_actor_model(input_shape, num_actions, learning_rate):
    X_inp = Input(shape = input_shape)
    advantages = Input(shape = [1])
    # X = Conv2D(32, 8, strides=(4,4), data_format = 'channels_first',
    #            activation = 'relu')(X_inp)
    # X = Conv2D(16, 4, strides=(2,2), data_format = 'channels_first', 
    #            activation = 'relu')(X)
    X = Flatten(input_shape=input_shape)(X_inp)
    X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(num_actions, activation = 'softmax')(X)

    def pg_loss(y_true, y_pred):
        clipped_y_pred = K.clip(y_pred, 1e-8, 1 - 1e-8)
        log_liklihood = y_true * K.log(clipped_y_pred)
        loss = K.sum(-log_liklihood * advantages)
        return loss
    
    model = Model(inputs = [X_inp, advantages], outputs = X)
    model.compile(optimizer = Adam(learning_rate = learning_rate), loss = pg_loss)

    prediction = Model(inputs=X_inp, outputs = X)

    return model, prediction

def get_critic_model(input_shape, learning_rate):
    X_inp = Input(shape = input_shape)
    # X = Conv2D(32, 8, strides=(4,4), data_format = 'channels_first',
    #            activation = 'relu')(X_inp)
    # X = Conv2D(16, 4, strides=(2,2), data_format = 'channels_first', 
    #            activation = 'relu')(X)
    X = Flatten(input_shape=input_shape)(X_inp)
    X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(1, activation = 'linear')(X)

    model = Model(inputs = X_inp, outputs = X)
    model.compile(optimizer = Adam(learning_rate = learning_rate), loss = Huber(delta = 1.5))

    return model


class A2CAgent(object):
    def __init__(self, env, train_flag = True, num_episodes = 20000, actor_learning_rate = 0.00025, 
                critic_learning_rate = 0.00025, gamma = 0.99, model_path = None, num_checkpoints = 10):
        self.env = env
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.num_checkpoints = num_checkpoints

        self.LEFT_ACTION = 2
        self.RIGHT_ACTION = 3
        self.action_space = [self.LEFT_ACTION, self.RIGHT_ACTION]
        
        self.num_actions = len(self.action_space)
        # self.num_actions = self.env.n_action
        self.model_path = model_path

        if(train_flag):
            self.actor_model, self.prediction = get_actor_model(self.env.observation_shape, self.num_actions, self.actor_learning_rate)
            self.critic_model = get_critic_model(self.env.observation_shape, self.critic_learning_rate)
        else:
            assert model_path!=None, "Please pass path model_path"
            self.prediction = load_model(model_path)
    
    def get_discounted_rewards(self, reward, gamma):
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0:
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        # Normalizing the discounted rewards
        discounted_r -= np.mean(discounted_r) 
        discounted_r /= np.std(discounted_r) 
        return discounted_r
    
    def train(self, render = False):
        all_episode_scores = []
        best_score = float('-inf')
        for episode in range(self.num_episodes):
            states = []
            actions = []
            rewards = []
            state = self.env.reset()
            episode_score = 0
            t = 0
            while(True):
                if(render):
                    self.env.render()
                action_probabilities = self.prediction.predict(state)[0]
                action = np.random.choice(range(self.num_actions), p = action_probabilities)
                next_state, reward, done, info = self.env.step(self.action_space[action])
                states.append(state)
                ohe_action = np.zeros((self.num_actions), dtype = np.float64)
                ohe_action[action] = 1
                actions.append(ohe_action)
                rewards.append(reward)
                
                state = next_state
                episode_score = episode_score + reward
                t = t + 1
                if(done or t>10000):
                    all_episode_scores.append(episode_score)
                    print("Episode {}/{} | Episode score : {} ({:.4})".format(episode+1, self.num_episodes, episode_score, np.mean(all_episode_scores[-50:])))
                    if( np.mean(all_episode_scores[-50:]) > best_score):
                        best_score = np.mean(all_episode_scores[-50:])
                        self.prediction.save(self.model_path)
                        print('Model Saved!')
                    break
            states_batch = np.vstack(states)
            actions_batch = np.vstack(actions)
            discounted_rewards = self.get_discounted_rewards(rewards, self.gamma)
            values = self.critic_model.predict(states_batch)[:, 0]
            advantages = discounted_rewards - values
            self.actor_model.train_on_batch([states_batch, advantages], actions_batch)
            self.critic_model.train_on_batch(states_batch, discounted_rewards)
            self.env.close()
            if(self.num_checkpoints != 0 and (episode % (self.num_episodes/self.num_checkpoints)) == 0):
                self.prediction.save('./saved_models/a2c-{:06d}.model'.format(episode))

    def test(self, render = True):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            episode_score = 0
            while(True):
                if(render):
                    self.env.render()
                    time.sleep(0.001)
                action_probabilities = self.prediction.predict(state)[0]
                action = np.argmax(action_probabilities)
                next_state, reward, done, info = self.env.step(self.action_space[action])
                state = next_state
                episode_score = episode_score + reward
                if(done):
                    print("Episode {}/{} | Episode score : {}".format(episode+1, self.num_episodes, episode_score))
                    break
            self.env.close()


    