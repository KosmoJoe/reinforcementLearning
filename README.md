# reinforcementLearning

In this project, I tested the openAI gym environments for reinforcement learning. Therefore, I implemented an abstract agent class which can act and learn (update Q-table, NN, ...) depending on an observation (state of the environment) from the environment.
For a great introduction into reinforcement learning, see the [Tutorials](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) by Arthur Juliani.

## Agents

Using this abstract definition of an agent, I implemented the four most common (and simple) agents:
1. A random acting agent: The simplest agent. Independent on the observation it performs a random action and does not learn.

2. An agent exploiting a Q-Tabular:
    *  Q-table is a list of values for every state (row) and action (column) possible in the environment. Its values answer the question how good it is to take a given action within a given state. Hence, observing state *s* the action one should take is given by *a = argmax(Q(s,:))*. Usually, one adds some randomness to this predicted action in the learning process to greedily pic an action.
    *  Learning is done using the Bellman equation, *Q(s,a) = r + gamma\*max(Q(s',:))* where *r* is the reward for the action taken and *s'* is the state which follows the action a starting from observation *s*. Here *gamma* is the so-called discount factor which is used to take care of the future possible reward.
    * In order to allow for continuous observation space models, a version of the agent where binning is used is implemented, too.
    
3. An agent with a neural network: Here a neural network is used to learn an approximation to a function which takes an observation (state) and predicts the q-value for all possible actions. Note that in contrast to the q-table approach, one can here also learn to act in a continuous state space. 
    * Update and learning strategy are the same as for the q-table approach.
    * The loss which is minimized during training is defined as *L = sum((Q_t - Q)^2)* where the target Q-value *Q_t* is computed as given above by the Bellman equation and *Q* is the output of the network for the state *s*. 
    
4. A DQN (deep q-network) approach to the reinforcement learning (be aware that for simplicity only a single layer is used). Extending on the above idea to use a neural network to learn a q-value function, here two further ideas are added:
    * Experiences Buffer: Storing the past experiences in the form of (observation,action,reward,newObservation,done) in a buffer allows to train the neural network not only on the immediate state and action but also to take the past into account. Drawing a random sample from the buffer further prevents from learning only from the immediate past but from a large set of experiences. In this way, one usually does not train every step taken in the environment but when training is executed one trains on a batch of experiences.
    * Separate (2nd) target network: In order to stabilize the training process, one uses a second neural network which represents the target q-values used to compute the loss during the training process. This migrates the risk of using constantly changing target values to adjust the network by updating the target q-network only on a much slower time scale.  


## Testing the Agents
In order to test the implemented agents, the openAI gym environments are used. In particular, a discreet environment, the FrozenLake, and a continuous environment, the cart-pole, are used.

1. FrozenLake environment 
   1. using the Q-tabular agent. Average Reward after 100 Episodes: 0.655
     ![Learning Curves](/img/frozenLake_Qtab.png)
   1. using neural network. Average Reward after 100 Episodes: 0.484
     ![Learning Curves](/img/frozenLake_QNN.png)
     
2. Cart-pole environment
   1. using a simple neural network agent. Average Reward after 100 Episodes: 175.3
     ![Learning Curves](/img/cartpole_QNN.png)
   2. using a double DQN with experience buffer. Average Reward after 100 Episodes: 199.348
       * Note that it is useful to keep a certain percentage of bad endings in the experience buffer (good hint by [Ironhead](http://blog.ironhead.ninja/2016/09/08/openai-cartpole.html)). Otherwise, the network would be over-trained on intermediate steps of the cart-pole game forgetting to stay away from the ending. 
       * Moreover, it seems important to punish the observation that the pole has fallen with high negative reward. 
     ![Learning Curves](/img/cartpole_DQN.png)
