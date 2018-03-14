##  Copyright (C) 2018 J. Schurer
##  This file is for testing the OpenAI gym environments


import numpy as np
import scipy as sc
from agents.agent import Agent


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space,observation_space,params):
        # Use uper init
        Agent.__init__(self,action_space,observation_space,params)
        
        # Set learning parameters
        self.episode_count = self.params[0]  # Number of episodes

    def _act(self, observation):
        return self.action_space.sample()

    def _update(self, observation, newObservation, action, reward):
        pass

if __name__ == '__main__':
    import gym
    env = gym.make('FrozenLake-v0')
    agent = RandomAgent(env.action_space,env.observation_space, [1000])