##  Copyright (C) 2018 J. Schurer
##  This file is for testing the OpenAI gym environments



import gym
from gym import wrappers
from agents.qtableAgent import QTableAgent
from agents.qNNAgent import QNNAgent
import matplotlib.pyplot as plt
from pylab import *

def low_pass(data, alpha=0.99):
    low_pass = [data[0]]
    for i in range(1,len(data)):
        low_pass.append(alpha*low_pass[-1] + (1.0-alpha)*data[i] )
    return low_pass
    


if __name__ == '__main__':
    # Debug Flags
    _debugOut = False 
    _debugPlot = False

    ###### Beginn   #####
    # Make Evironment
    env = gym.make('FrozenLake-v0')
    #method = 0
    #env.render(mode='human', close=False)

    ###### TRAIN  #####
    episode_count = 2000
    #method = 0
    #learnRates = np.linspace(0.75,0.85, num=3,endpoint=False)
    #discounts = np.linspace(0.94,0.96, num=3,endpoint=False)
    method = 1
    learnRates = [0.2, 0.2, 0.2, 0.3, 0.3, 0.3] #np.linspace(0.2,0.4, num=3,endpoint=True)
    discounts = [0.99]  #np.linspace(0.99,0.999, num=1,endpoint=True)
    rewards = []
    steps = []
    reward = 0
    XlearnRate = 0
    Xdiscount = 0
    for j in learnRates:
        for k in discounts:
            params = [episode_count, j,k,0.1]
            if method == 0:
                agent = QTableAgent(env.action_space,env.observation_space, params)
            elif method == 1:
                agent = QNNAgent(env.action_space,env.observation_space, params,discreet=True)
            else:
                raise ValueError
            agent._render = False
            rew, step = agent.train(env)
            rewards.append(rew)
            steps.append(step)
            if reward < sum(rewards[-1])/episode_count:
                reward = sum(rewards[-1])/episode_count
                XlearnRate = j
                Xdiscount = k
                mytrainedAgent = agent
            plt.plot(low_pass(rewards[-1]),label=("Test: "+str(j+k)))


    plot([0.78]*len(rewards[-1]), 'g', label="Pass Mark")
    plt.show()
    print("Maximal Reward: " + str(reward) + "  at lr=" +str(XlearnRate) + "  and dis=" +str(Xdiscount))

    ###### TEST   #####
    reward = []
    mytrainedAgent.episode_count = 100
    for kk in range(10):
        rew, step = mytrainedAgent.test(env)
        reward.append(sum(rew))
    print("Average Reward after 100 Episodes: " + str(sum(reward)/(100*10)))

    ###### END   #####
    # Close the env and write monitor result info to disk
    env.close()

    


