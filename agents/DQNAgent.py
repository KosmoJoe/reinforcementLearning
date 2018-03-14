##  Copyright (C) 2018 J. Schurer
##  This file is for testing the OpenAI gym environments


import numpy as np
import scipy as sc
import random
from agents.agent import Agent
import tensorflow as tf
import tensorflow.contrib.slim as slim

class DQNAgent(Agent):
    """A Deep Q-Network Agent using a neural network, an experience buffer and an additional traget network"""
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
        self.discount = self.params[2]   # Time range value for reward
        self.epsi = self.params[3]   # Epsilon for greedy picking
        self.epsi_decay = self.params[4]
        
        self.pretrainEpi = 250   # Number of steps before first train
        self.batch_size = 200 #Size of training batch
        self.trainPadding = 5 # Every xth step a training occurs
        self.tau = 0.01  #Amount to update target network at each step.
        self.method = self.selectMethod("e-greedy")
        
        self.epsi_min = 0.001
        
        self.currEpisode = 0 # Current training stage epsiode
        self.time = 0 # Current frame within one episode
        self._timeTot = 200 # Maximal time in one episode
        self.currQs = None # Current prediction for the Q values using current observation

        tf.reset_default_graph()
        self.qNet = Q_Network([[self.inputN, 128, self.actionN],self.learnRate])
        self.targetQNet = Q_Network([[self.inputN, 128, self.actionN],self.learnRate])
        self.myBuffer = ExperienceBuffer()
        
        init = tf.global_variables_initializer()
        trainables = tf.trainable_variables()
        self.targetOps = Q_Network.updateTargetGraph(trainables,self.tau)
        self.session = tf.Session()
        self.session.run(init)


    def _act(self, observation):
        self.time += 1
        if self.discreet:
            observation = self.transformObservation(observation)
        #Choose an action by greedily (with e chance of random action) from the Q-network
        return self.method(observation)

    def _update(self, observation, newObservation, action, reward, done):
        if self.discreet:
            observation = self.transformObservation(observation)
            newObservation = self.transformObservation(newObservation)
        
        reward = self.time #use cumulative reward
        
        if done:
            if self.time + 1 < self._timeTot:
                reward = -500.0  # Penalize bad endings
            self.time = 0
            
        #Update Q-Table with new knowledge 
        self.myBuffer.add(np.reshape(np.array([observation,action,reward,newObservation,done]),[1,5]))
        
        if self.currEpisode > self.pretrainEpi and self.currEpisode % self.trainPadding == 0:
        #if self.currEpisode > self.pretrainEpi and done:
            #We use Double-DQN training algorithm
            trainBatch = self.myBuffer.sample(self.batch_size)
            actionPred,allQ = self.session.run([self.qNet.predict,self.qNet.Qout],feed_dict={self.qNet.inputs:np.vstack(trainBatch[:,3]),self.qNet.keep_per:1.0})
            Q2 = self.session.run(self.targetQNet.Qout,feed_dict={self.targetQNet.inputs:np.vstack(trainBatch[:,3]),self.targetQNet.keep_per:1.0})
            
            end_multiplier = -(trainBatch[:,4] - 1)
            doubleQ = np.amax(Q2,1)
            #doubleQ = Q2[range(self.batch_size),actionPred]
            targetQ = trainBatch[:,2] + self.discount*doubleQ * end_multiplier
            
            if False:  # To test without target qnet
                targetQ = trainBatch[:,2] + self.discount*np.amax(allQ,1)*end_multiplier
                
            _ = self.session.run(self.qNet.updateModel,feed_dict={self.qNet.inputs:np.vstack(trainBatch[:,0]),self.qNet.nextQ:targetQ,self.qNet.keep_per:1.0,self.qNet.actions:trainBatch[:,1]})
            Q_Network.updateTarget(self.targetOps,self.session)

    def transformObservation(self, observation):
        return np.identity(self.inputN)[observation:observation+1]


    def _saveModel(self, checkpointpath):
        saver = tf.train.Saver()
        save_path = saver.save(self.session, checkpointpath + ".ckpt", global_step=self.currEpisode)

    
    def _restoreModel(self, checkpointpath):
        saver = tf.train.Saver()
        saver.restore(self.session, checkpointpath)

    #########################################################################
    ## Methods of how to pick the action from the last observation
    def _greedy(self,observation):
        action,allQ = self.session.run([self.qNet.predict,self.qNet.Qout],feed_dict={self.qNet.inputs:[observation],self.qNet.keep_per:1.0})
        return action[0]

    def _egreedy(self,observation):
        action,allQ = self.session.run([self.qNet.predict,self.qNet.Qout],feed_dict={self.qNet.inputs:[observation],self.qNet.keep_per:1.0})
        action = action[0]
        self.currEpsi = self.epsi if self.currEpisode < self.pretrainEpi else self.epsi*np.exp(-self.epsi_decay*(self.currEpisode-self.pretrainEpi))
        #currEpsi = self.epsi*np.exp(-self.epsi_decay*self.currEpisode)
        #if self.currEpisode%100 == 0:
        #    print(currEpsi)
        if np.random.rand(1) < max(self.currEpsi, self.epsi_min): # self.epsi/(self.currEpisode/50 + 10):  #
            action = np.argmax(np.random.randn(1,self.action_space.n))
        return action

    def _boltzmann(self,observation):
        Qd,allQ = self.session.run([self.qNet.Qdist,self.qNet.Qout],feed_dict={self.qNet.inputs:[observation],self.qNet.Temp:e,self.qNet.keep_per:1.0})
        action = np.random.choice(Qd[0],p=Qd[0])
        return np.argmax(Qd[0] == action)

    def _bayesian(self,observation):
        action,allQ = self.session.run([self.qNet.predict,self.qNet.Qout],feed_dict={self.qNet.inputs:[observation],self.qNet.keep_per:(1-self.epsi/((self.currEpisode-self.pretrainEpi)/50 + 10))+0.1})
        return action[0]



class Q_Network():
    """ Class  which defines the neural network for the Agent """
    def __init__(self,params):

        self.tree = params[0]  # [numIn, numHidden, numOut]
        self.learnRate = params[1]
    
        #Feed-forward part
        self.inputs = tf.placeholder(shape=[None,self.tree[0]],dtype=tf.float32)
        self.keep_per = tf.placeholder(shape=None,dtype=tf.float32)
        self.Temp = tf.placeholder(shape=None,dtype=tf.float32)
        
        hidden1 = slim.fully_connected(self.inputs,self.tree[1],activation_fn=tf.nn.relu,weights_initializer=tf.random_uniform_initializer(-1.0, 1.0), biases_initializer=tf.random_uniform_initializer(-1.0, 1.0))
        dropout1 = slim.dropout(hidden1,self.keep_per)
        hidden2 = slim.fully_connected(dropout1,128,activation_fn=tf.nn.relu,weights_initializer=tf.random_uniform_initializer(-1.0, 1.0),biases_initializer=tf.random_uniform_initializer(-1.0, 1.0))
        dropout2 = slim.dropout(hidden2,self.keep_per)
        self.Qout = slim.fully_connected(dropout2,self.tree[2],activation_fn=None,weights_initializer=tf.random_uniform_initializer(-1.0, 1.0),biases_initializer=tf.random_uniform_initializer(-1.0, 1.0))
        self.predict = tf.argmax(self.Qout,1)
        self.Qdist = tf.nn.softmax(self.Qout/self.Temp)

        #Loss function to predict Q-Values
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,self.tree[2],dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)
        
        self.nextQ = tf.placeholder(shape=[None],dtype=tf.float32)
        loss = tf.reduce_mean(tf.square(self.nextQ - self.Q))
        
        #trainer = tf.train.GradientDescentOptimizer(learning_rate=self.learnRate)
        trainer = tf.train.AdamOptimizer(learning_rate=self.learnRate)
        self.updateModel = trainer.minimize(loss)
        
        self._tf_session = tf.InteractiveSession()
        self._tf_session.run(tf.initialize_all_variables())
    
    @classmethod
    def updateTargetGraph(cls,tfVars,tau):
        total_vars = len(tfVars)
        #print(total_vars)
        #print(tfVars)
        op_holder = []
        for idx,var in enumerate(tfVars[0:int(total_vars/2)]):
            op_holder.append(tfVars[idx+int(total_vars/2)].assign((var.value()*tau) + ((1-tau)*tfVars[idx+int(total_vars/2)].value())))
        return op_holder

    @classmethod
    def updateTarget(cls,op_holder,sess):
        for op in op_holder:
            sess.run(op)
        
        
                
class ExperienceBuffer():
    """ Class to store and retrieve Experiences from an Agent"""
    def __init__(self, buffer_size = 10000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.idx = 0
        self.totDone = 0
        self.endRatio = 0.1
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            if self.totDone <= self.endRatio*self.buffer_size: 
                idx=next(i for i,x in enumerate(self.buffer) if not x[4])   #Find first non done entry
            else:
                idx=next(i for i,x in enumerate(self.buffer) if x[4])   #Find first done entry
                self.totDone -=1 
            self.buffer[idx:idx+1] = []
                    
        if experience[0,4]:
            self.totDone += 1 
        self.buffer.extend(experience)

            
    def sample(self,SampleSize):
        return np.reshape(np.array(random.sample(self.buffer,SampleSize)),[SampleSize,5])

if __name__ == '__main__':
    import gym
    envCont = gym.make('CartPole-v0')
    agent = QNNAgent(envCont.action_space,envCont.observation_space, [1000, 0.9, 0.8, 0.1])