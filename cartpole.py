##  Copyright (C) 2018 J. Schurer
##  This file is for testing the OpenAI gym environments


import gym
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from agents.qNNAgent import QNNAgent
from pylab import *

def low_pass(data, alpha=0.99):
    low_pass = [data[0]]
    for i in range(1,len(data)):
        low_pass.append(alpha*low_pass[-1] + (1.0-alpha)*data[i] )
    return low_pass


if __name__ == '__main__':
    # Make Evironment
    env = gym.make('CartPole-v0')

    # Make Agentparams = [episode_count, 0.8,0.8, 0.1]
    episode_count = 2000
    learnRates = [1e-3] #[ 1e-4,1e-3,1e-2]#np.linspace(0.0001,0.001, num=3,endpoint=False)
    discounts = [0.95]#[ 0.90, 0.95, 0.99] #[0.95, 0.97, 0.99]#np.linspace(0.94,0.96, num=3,endpoint=False)
    epsi = [1.0] #[ 0.1, 1.0]
    epsi_decay  = [0.01]#[0.001, 0.005, 0.01]
    reward = 0
    XlearnRate = 0
    Xdiscount = 0
    Xepsi = 0
    Xdec = 0
    rewards = []
    legs = []
    for j in learnRates:
        for k in discounts:
            for ep in epsi:
                for epdec in epsi_decay:
                    params = [episode_count, j,k, ep, epdec]

                    agent = QNNAgent(env.action_space,env.observation_space, params)
                    agent._render = False
    
                    # Train Agent
                    rewardList, stepList =  agent.train(env)
                    rewards.append(rewardList)
                    if reward < sum(rewardList[-1])/episode_count:
                        reward = sum(rewardList[-1])/episode_count
                        mytrainedAgent = agent
                        XlearnRate = j
                        Xdiscount = k
                        Xepsi = ep
                        Xdec = epdec
                    p = plt.plot(low_pass(rewardList),label=("Test: "+str(params)))
                    legs.append(p)

    pm = plot([195]*len(rewards[-1]), 'g', label="Pass Mark")
    legs.append(pm)
    plt.legend()
    plt.show()
    print("Maximal Reward: " + str(reward) + "  at lr=" +str(XlearnRate) + "  and dis=" +str(Xdiscount) + "  and epsi=" +str(Xepsi) + "  and epsi_dec=" +str(Xdec))

    
        ###### TEST   #####
    reward = []
    mytrainedAgent.episode_count = 100
    mytrainedAgent.epsi = mytrainedAgent.epsi_min
    for kk in range(10):
        rew, step = mytrainedAgent.test(env)
        reward.append(sum(rew))
    print("Average Reward after 100 Episodes: " + str(sum(reward)/(100*10)))
    
    #print("Maximal Reward: " + str(reward) + "  at lr=" +str(XlearnRate) + "  and dis=" +str(Xdiscount))