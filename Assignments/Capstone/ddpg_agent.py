from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model, save_model
import tensorflow as tf
import numpy as np

from oua_action_noise import OUActionNoise
from replay_memory import Memory 

def get_actor(states_shape, n_actions, upper_bound):
    X_inp = Input(shape = (states_shape, ))
    X = Dense(128, activation="relu")(X_inp)
    # X = BatchNormalization()(X)
    X = Dense(128, activation="relu")(X)
    # X = BatchNormalization()(X)
    X = Dense(n_actions, activation="tanh")(X)
    X = X * upper_bound
    return Model(X_inp, X)

def get_critic(states_shape, action_shape):
    X_inp_states = Input(shape = (states_shape))
    # X_states = Dense(32, activation='relu')(X_inp_states)
    # X_states = Dense(32, activation='relu')(X_states)

    X_inp_actions = Input(shape = (action_shape))
    # X_actions = Dense(32, activation='relu')(X_inp_actions)
    # X_actions = Dense(32, activation='relu')(X_actions)

    X = Concatenate()([X_inp_states, X_inp_actions])
    # X = BatchNormalization()(X)
    X = Dense(128, activation="relu")(X)
    # X = BatchNormalization()(X)
    X = Dense(128, activation="relu")(X)
    # X = BatchNormalization()(X)
    X = Dense(1)(X)
    return Model([X_inp_states, X_inp_actions], X)

class DDPGAgent:
    def __init__(self, env, train_flag = True, model_path = None, actor_learning_rate = 0.001, critic_learning_rate = 0.002,
                num_episodes = 1000, tau = 0.005, gamma = 0.99, memory_size = 100000, batch_size = 64):
        self.env = env
        self.train_flag = train_flag
        self.num_episodes = num_episodes
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.model_path = model_path

        self.actor_opt = Adam(lr = actor_learning_rate)
        self.critic_opt = Adam(lr = critic_learning_rate)

        self.states_shape = self.env.observation_space.shape[0]
        self.action_shape = self.env.action_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.actions_lower_bound = env.action_space.low
        self.actions_upper_bound = env.action_space.high

        if self.train_flag:
            self.noise = OUActionNoise(mean = np.zeros(self.n_actions), std_deviation = 0.2 * np.ones(self.n_actions))
            self.actor = get_actor(self.states_shape, self.n_actions, self.actions_upper_bound)
            self.actor_target = get_actor(self.states_shape, self.n_actions, self.actions_upper_bound)
            self.critic = get_critic(self.states_shape, self.action_shape)
            self.critic_target = get_critic(self.states_shape, self.action_shape)

            self.actor_target.set_weights(self.actor.get_weights())
            self.critic_target.set_weights(self.critic.get_weights())
            self.memory = Memory(memory_size)
        else:
            self.actor = load_model(self.model_path)
        
    def get_action(self, state):
        state = tf.cast(tf.expand_dims(tf.convert_to_tensor(state), 0), tf.float32)
        actions = tf.squeeze(self.actor(state)).numpy()
        if self.train_flag:
            noise = self.noise
            actions = actions + noise()

        actions = np.clip(actions, self.actions_lower_bound, self.actions_upper_bound)
        return actions

    def update_target_weights(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def learn(self):
        batch, batch_size = self.memory.get_batch(self.batch_size)
        state_batch = np.zeros((batch_size, self.states_shape))
        action_batch = np.zeros((batch_size, self.action_shape))
        reward_batch = np.zeros((batch_size, 1))
        next_state_batch = np.zeros((batch_size, self.states_shape))
        done_batch = np.zeros(batch_size)

        for i, record in enumerate(batch):
            state_batch[i] = record['state']
            action_batch[i] = record['action']
            reward_batch[i] = record['reward']
            next_state_batch[i] = record['next_state']
        
        # print(reward_batch.shape)
        state_batch = tf.convert_to_tensor(state_batch, dtype = tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype = tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype = tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype = tf.float32)

        # Critic Update
        with tf.GradientTape() as g:
            target_actions = self.actor_target(next_state_batch)
            y = reward_batch + self.gamma * self.critic_target(
                [next_state_batch, target_actions]
            )
            critic_value = self.critic([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_gradient = g.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))

        # Actor Update
        with tf.GradientTape() as g:
            actions = self.actor(state_batch)
            critic_value = self.critic([state_batch, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)
        
        actor_gradient = g.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        # Soft update target network
        self.update_target_weights(self.actor_target.variables, self.actor.variables, self.tau)
        self.update_target_weights(self.critic_target.variables, self.critic.variables, self.tau)

    def train(self, render = False):
        all_episode_scores = []
        best_score = float('-inf')
        for episode in range(self.num_episodes):
            state = self.env.reset()
            episode_score = 0
            t = 0
            while(True):
                if(render):
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.memory.add_record(state, action, reward, next_state, done)
                self.learn()
                state = next_state
                episode_score = episode_score + reward
                t = t + 1
                if(done or t>1000):
                    all_episode_scores.append(episode_score)
                    print("Episode {}/{} | Episode score : {} ({:.4})".format(episode+1, self.num_episodes, episode_score, np.mean(all_episode_scores[-50:])))
                    if( np.mean(all_episode_scores[-50:]) > best_score):
                        best_score = np.mean(all_episode_scores[-50:])
                        if self.model_path is not None:
                            self.actor.save(self.model_path)
                            print('Model Saved!')
                    break
            self.env.close()

    def test(self, render = True):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            episode_score = 0
            while(True):
                if(render):
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                episode_score = episode_score + reward
                if(done):
                    print("Episode {}/{} | Episode score : {}".format(episode+1, self.num_episodes, episode_score))
                    break
            self.env.close()