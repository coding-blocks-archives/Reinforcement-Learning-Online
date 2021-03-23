import numpy as np
import gym
import cv2

class PongEnv(object):
    def __init__(self, env_name, frame_size = (80,80), 
                 binarize = True, stack_size = 4):
        self.env = gym.make(env_name)
        self.frame_size = frame_size
        self.stack_size = stack_size
        self.binarize = binarize
        self.frame_stack = np.zeros((self.stack_size, self.frame_size[0], self.frame_size[1]), dtype = np.float64)
        self.n_action = self.env.action_space.n
        self.observation_shape = self.frame_stack.shape
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        processed_obs = self.process_obs(obs)
        return processed_obs, reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        for i in range(20):
            obs, _, _, _ = self.env.step(0)
        self.frame_stack = np.zeros((self.stack_size, self.frame_size[0], self.frame_size[1]))
        processed_obs = self.process_obs(obs)
        return processed_obs
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
    
    def process_obs(self, obs):
        clip_obs = obs[35:195:2,::2,:]
        grayscale_obs = cv2.cvtColor(clip_obs, cv2.COLOR_RGB2GRAY)
        if(grayscale_obs.shape != self.frame_size):
            grayscale_obs = cv2.resize(grayscale_obs, self.frame_size,            
                                    interpolation=cv2.INTER_CUBIC)
        if(self.binarize):
            grayscale_obs[grayscale_obs < 100] = 0
            grayscale_obs[grayscale_obs >= 100] = 255.0
            
        grayscale_obs = grayscale_obs.astype(np.float64) / 255.0
        self.frame_stack = np.roll(self.frame_stack, shift = 1, axis = 0)
        self.frame_stack[0,:,:] = grayscale_obs
        return np.expand_dims(self.frame_stack, 0)