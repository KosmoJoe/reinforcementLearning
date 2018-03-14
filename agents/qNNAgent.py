##  Copyright (C) 2018 J. Schurer
##  This file is for testing the OpenAI gym environments


import numpy as np
import scipy as sc
from agents.agent import Agent
import tensorflow as tf


class QNNAgent(Agent):
    """A Q-Network Agent using a neural network"""
    def __init__(self, action_space,observation_space,params,discreet=False):
        # Use uper init
        Agent.__init__(self,action_space,observation_space,params)
        self.discreet = discreet
        if discreet:
            self.inputN = self.observation_space.n
        else:
            self.inputN = self.observation_space.shape[0]
        self.actionN = self.action_space.n
        
        # Set learning parameters
        self.episode_count = self.params[0]  # Number of episodes
        self.learnRate = self.params[1]  # Number of episodes
        self.dicount = self.params[2]   # Time range value for reward
        self.epsi = self.params[3]   # Epsilon for greedy picking
        self.epsi_decay = self.params[4]
        self.epsi_min = 0.001
        #define TF graph
        tf.reset_default_graph()
        #graph1 = tf.Graph()
        #with graph1.as_default():
        #These lines establish the feed-forward part of the network used to choose actions
        
        n_hidden_1  = 64
        n_hidden_2  = 32
        self.inputs1 = tf.placeholder(shape=[1,self.inputN],dtype=tf.float32)
        #W1 = tf.Variable(tf.random_uniform([self.inputN,self.actionN],0,0.01))
        
        W1 = tf.Variable(tf.random_normal([self.inputN,n_hidden_1]))
        W2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
        W3 = tf.Variable(tf.random_normal([n_hidden_2, self.actionN]))
        
        layer_1 = tf.nn.relu(tf.matmul(self.inputs1, W1))
        layer_2 = tf.nn.relu(tf.matmul(layer_1, W2))
        self.Qout = tf.matmul(layer_2, W3)
        
        #self.Qout = tf.matmul(self.inputs1,self.W)
        self.predict = tf.argmax(self.Qout,1)
        
        self.currEpisode = 0 # Current training stage epsiode
        self.currQs = None # Current prediction for the Q values using current observation

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.placeholder(shape=[1,self.actionN],dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        trainer = tf.train.AdamOptimizer(learning_rate=self.learnRate)
        #trainer = tf.train.GradientDescentOptimizer(learning_rate=self.learnRate)
        self.updateModel = trainer.minimize(loss)
        
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def _act(self, observation):
        if self.discreet:
            observation = self.transformObservation(observation)
        #Choose an action by greedily (with e chance of random action) from the Q-network
        action,allQ = self.session.run([self.predict,self.Qout],feed_dict={self.inputs1:observation.reshape(1,self.inputN)})
        self.currQs = allQ
        action = action[0]
        
        self.currEpsi = self.epsi*np.exp(-self.epsi_decay*self.currEpisode)
        if np.random.rand(1) < max(self.currEpsi, self.epsi_min): # self.epsi/(self.currEpisode/50 + 10):  #
        #if np.random.rand(1) < self.epsi/(self.currEpisode/50 + 1):
            action = np.argmax(np.random.randn(1,self.action_space.n))
        return action

    def _update(self, observation, newObservation, action, reward, done):
        if self.discreet:
            observation = self.transformObservation(observation)
            newObservation = self.transformObservation(newObservation)
        # Penalize endings
        reward = reward if not done else -10
        
        #if done:
        #     print("episode: {}/{}, score: {}, e: {:.2}".format(self.currEpisode, self.episode_count, reward, self.currEpsi))
        #Update Q-Table with new knowledge 
        QatNewObs = self.session.run(self.Qout,feed_dict={self.inputs1:newObservation.reshape(1,self.inputN)})
        targetQ = self.currQs
        targetQ[0,action] = reward if done else reward + self.dicount*np.max(QatNewObs)
        #_,W1 = self.session.run([self.updateModel,self.W],feed_dict={self.inputs1:observation.reshape(1,self.inputN),self.nextQ:targetQ})
        self.session.run([self.updateModel],feed_dict={self.inputs1:observation.reshape(1,self.inputN),self.nextQ:targetQ})

    def transformObservation(self, observation):
        return np.identity(self.inputN)[observation:observation+1]


if __name__ == '__main__':
    import gym
    envCont = gym.make('CartPole-v0')
    agent = QNNAgent(envCont.action_space,envCont.observation_space, [1000, 0.9, 0.8, 0.1])