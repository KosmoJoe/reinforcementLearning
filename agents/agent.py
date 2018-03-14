##  Copyright (C) 2018 J. Schurer
##  This file is for testing the OpenAI gym environments



class Agent(object):
    """The abstract definition of the agent class. It defines an agent and the asociated
    functions. 
    The main API methods that users of this class need to know are:
        act()
        update()
    When implementing an agent, override the following methods
    in your subclass:
        _act
        _update
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
    Note: xx
    """
    ## Some debug flags
    _debugOut = True
    _debugPlot = False

    
    def __init__(self, action_space,observation_space,params):
        """Main Agent Initialization
        Args:
            action_space (object): the action space (OpenAI.gym)
            observation_space (object): the observation space (OpenAI.gym)
            params (object): list of parameters for the agent
        Returns:
            action (object): action to take 
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.params = params
        self.episode_count = params[0]
        self.currEpisode = 0
        self._maxLoop = 10000
        self._render = True
        self.method = self.selectMethod("greedy")
        
        self.saveStepping = 10000   # every episode%saveStepping==0  
        self.checkpointpath = './tmp/model'
        
    # Override in ALL subclasses
    def _act(self, observation): raise NotImplementedError
    def _update(self,observation, newObservation, action, reward, done): raise NotImplementedError
    def _saveModel(self, checkpointpath): raise NotImplementedError
    def _restoreModel(self, checkpointpath): raise NotImplementedError

    
    def act(self, observation):
        """Choose an action depending on observation
        Accepts an observation and return an action.
        Args:
            observation (object): an observation provided by the environment (OpenAI.gym)
        Returns:
            action (object): action to take 
        """
        return self._act(observation)

    def update(self,  observation, newObservation, action, reward, done):
        """Update the agents action selection method to learn.
        Accepts an observation and return an action.
        Args:
            observation (object): an observation provided by the environment (OpenAI.gym)
        Returns:
            NONE
        """
        self._update(observation, newObservation, action, reward, done)


    #################################
    ######  Load / Save
    #################################

    def saveModel(self, checkpointpath):
        """Save the parameter of an agent at the current stage
        Accepts an path including the model name
        Args:
            checkpointpath (str): the path where to save the model including the model save name
        Returns:
            NONE
        """
        self._saveModel(checkpointpath)

    def restoreModel(self, checkpointpath):
        """Load the parameter of an agent from a file
        Accepts an path including the model name
        Args:
            checkpointpath (str): the path where to load the model including the model save name
        Returns:
            NONE
        """
        self._restoreModel(checkpointpath)


    #################################
    ######  Test / Train
    #################################

    def train(self, env):
        """Train an agent with the parameters params in the environment env.
        Args:
            env (gym.environment): the environment (OpenAI.gym)
        Returns:
            rewardList (list): reward per episode
            stepList (list): steps per episode 
        """
        return self._testOrTrain(env, train=True)
        
    def test(self, env):
        """Test an agent with the parameters params in the environment env.
        Args:
            env (gym.environment): the environment (OpenAI.gym)
        Returns:
            rewardList (list): reward per episode
            stepList (list): steps per episode 
        """
        return self._testOrTrain(env, train=False)


    def _testOrTrain(self, env, train=True):
        rewardList = []
        stepList = []
        done = False

        for i in range(self.episode_count):
            observation = env.reset()
            self.currEpisode = i
            rewardSum = 0.0
            steps = 0
            while True:
                steps += 1
                
                action = self.act(observation)
                newObservation, reward, done, _ = env.step(action)
                
                rewardSum += reward
                if train: 
                    self.update(observation, newObservation, action, reward, done )

                
                observation = newObservation
                
                if self._render:
                    env.render(mode='human', close=False)
                if done:
                    break # One Episode is finished
                if steps > self._maxLoop:
                    break  # End loop
            rewardList.append(rewardSum)# + rewardList[-1])
            stepList.append(steps)

            if i>0 and i%100 == 0:
                print("#############   Episode: " + str(i)+ "    ###############")
                print("Average Reward: " + str(sum(rewardList[-100:])/100))
            #if Agent._debugOut:
            #    print("Reward: " + str(rewardSum))
            #    print("Steps: " + str(steps))

            if i>0 and train and i%self.saveStepping == 0:
                self.saveModel(self.checkpointpath)
                
        if Agent._debugOut:
            print("Score over time: " +  str(sum(rewardList)/self.episode_count))
        

        # Learning Curve
        if Agent._debugPlot:
            plt.plot(low_pass(rewardList))
            plt.show()
        return rewardList, stepList
    
    
    
    #####
    ## Methods of how to pick the action from the last observation
    def _random(self):
        return self.env.action_space.sample()

    def _greedy(self,observation): raise NotImplementedError
    def _egreedy(self,observation): raise NotImplementedError
    def _boltzmann(self,observation): raise NotImplementedError
    def _bayesian(self,observation): raise NotImplementedError

    def selectMethod(self,method):
        methods = {"greedy" : self._greedy,
                   "random" : self._random,
                 "e-greedy" : self._egreedy,
               " boltzmann" : self._boltzmann,
                 "bayesian" : self._bayesian}
        return methods[method]
    
if __name__ == '__main__':
    pass