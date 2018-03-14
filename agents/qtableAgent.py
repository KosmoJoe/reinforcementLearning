##  Copyright (C) 2018 J. Schurer
##  This file is for testing the OpenAI gym environments


import numpy as np
import scipy as sc
from agents.agent import Agent



class QTableAgent(Agent):
    """A standart Q-Table Agent"""
    def __init__(self, action_space,observation_space,params):
        # Use uper init
        Agent.__init__(self,action_space,observation_space,params)
        
        #Initialize table with all zeros
        self.Q = np.zeros([observation_space.n,action_space.n])

        # Set learning parameters
        self.episode_count = self.params[0]  # Number of episodes
        self.lr = self.params[1] #.5  # Learning Rate
        self.y = self.params[2]  # .8  # Discount Factor
        self.currEpisode = 0 # Current training stage epsiode

    def _act(self, observation):
        #Choose an action by greedily (with noise) picking from Q table
        return np.argmax(self.Q[observation,:] + np.random.randn(1,self.action_space.n)*(1./(self.currEpisode+1)))

    def _update(self, observation, newObservation, action, reward, done):
        #Update Q-Table with new knowledge    
        self.Q[observation,action] = self.Q[observation,action] + self.lr*(reward + self.y*np.max(self.Q[newObservation,:]) - self.Q[observation,action])


class QTableAgentOwn(Agent):
    """The most simple Q-Table Agent"""
    def __init__(self, action_space,observation_space,params):
        # Use uper init
        Agent.__init__(self,action_space,observation_space,params)
        
        #Initialize table with all zeros
        self.Q = np.zeros([observation_space.n,action_space.n])
        
        # Set learning parameters
        self.episode_count = self.params[0]  # Number of episodes

    def _act(self, observation):
        #Choose an action by picking random if not learned
        if np.max(self.Q[observation,:]) > 0:
            return np.argmax(self.Q[observation,:])
        else:
            return np.argmax(np.random.randn(1,self.action_space.n))

    def _update(self, observation, newObservation, action, reward, done):
        #Update Q-Table with new knowledge    
        self.Q[observation,action] = reward + np.max(self.Q[newObservation,:])

        
        
class QTableAgentBinned(Agent):
    """A Q-Table Agent for non-discrite observation space
    Defines bins to discretize the observation space and build-up a Q-Table
    """
    def __init__(self, action_space,observation_space,params):
        # Use uper init
        Agent.__init__(self,action_space,observation_space,params)
        
        # Set learning parameters
        self.episode_count = params[0]  # Number of episodes
        self.lr = params[1] #.5  # Learning Rate
        self.y = params[2]  # .8  # Discount Factor
        self.binsize = params[3]  # Should be uneven to distinguish -epsi and epsi
        self.currEpisode = 0 # Current training stage epsiode
        
        #Initialize table with all zeros
        self.Q = np.zeros([np.power(self.binsize,observation_space.shape[0]),action_space.n])
        
        # Determine Bins
        self.low = [-0.5, -2, -0.25, -2] #self.observation_space.low
        self.high = [0.5, 2, 0.25 ,2]  # self.observation_space.high
        self.createBins()


    def _act(self, observation):
        #Choose an action by greedily (with noise) picking from Q table
        idx = self.getIndex(observation)
        return np.argmax(self.Q[idx,:] + np.random.randn(1,self.action_space.n)*(1./(self.currEpisode+1)))

    def _update(self, observation, newObservation, action, reward, done):
        #Update Q-Table with new knowledge
        idx = self.getIndex(observation)
        self.Q[idx,action] = (1-self.lr)*self.Q[idx,action] + self.lr*(reward + self.y*np.max(self.Q[self.getIndex(newObservation),:]))

    def createBins(self):
        # Determine Bins
        self.bins = []
        for j in range(self.observation_space.shape[0]):
            bin = np.linspace(start=self.low[j], stop=self.high[j], num=self.binsize)
            self.bins.append(bin)

    def getIndex(self, observation):
        index = 0
        idx_tmp = []
        for b in range(self.observation_space.shape[0]):
            idx_tmp.append(np.argmin(abs(self.bins[b] - observation[b])))
            index += idx_tmp[b]*np.power(self.binsize,b)
        return index



if __name__ == '__main__':
    import gym
    envCont = gym.make('CartPole-v0')
    env = gym.make('FrozenLake-v0')
    
    agent = QTableAgent(env.action_space,env.observation_space, [1000, 0.9, 0.8])
    agent = QTableAgentOwn(env.action_space,env.observation_space, [1000])
    agent = QTableAgentBinned(envCont.action_space,envCont.observation_space, [1000, 0.9, 0.8, 5])