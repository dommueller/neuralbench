import gym
import tensorflow as tf 
import numpy as np 
import random
from collections import deque
from dqn_params import DqnParams

class DQN():
    # DQN Agent inspired by: Flood Sung
    # https://gist.github.com/songrotek/3b9d893f1e0788f8fad0e6b49cde70f1#file-dqn-py
    def __init__(self, env, net_size, hyperparameters, max_evaluations):
        self.hyperparameters = hyperparameters
        self.max_evaluations = max_evaluations
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.epsilon = self.hyperparameters.initial_epsilon
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n 

        tf.reset_default_graph()
        self.create_Q_network(net_size)
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def create_Q_network(self, net_size):
        # network weights
        W1 = self.weight_variable([self.state_dim,net_size])
        b1 = self.bias_variable([net_size])
        W2 = self.weight_variable([net_size,self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float",[None,self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer,W2) + b2

    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.action_dim])
        self.y_input = tf.placeholder("float",[None])
        Q_action = tf.reduce_sum(tf.mul(self.Q_value,self.action_input),reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
        if len(self.replay_buffer) > self.hyperparameters.replay_size:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > self.hyperparameters.batch_size:
            self.train_Q_network()

    def train_Q_network(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,self.hyperparameters.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        for i in range(0,self.hyperparameters.batch_size):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + self.hyperparameters.gamma * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
            })

    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            })[0]
        if random.random() <= self.epsilon:
            action = random.randint(0,self.action_dim - 1)
        else:
            action = np.argmax(Q_value)

        self.epsilon -= (self.hyperparameters.initial_epsilon - self.hyperparameters.final_epsilon)/self.max_evaluations
        return action

    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            })[0])

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

class DQN_continous(DQN):
    def __init__(self, env, net_size, hyperparameters, max_evaluations):
        self.hyperparameters = hyperparameters
        self.max_evaluations = max_evaluations
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.epsilon = self.hyperparameters.initial_epsilon
        self.state_dim = len(np.reshape(env.observation_space.sample(), -1))
        self.action_dim = 2 * len(np.reshape(env.action_space.sample(), -1))

        self.create_Q_network(net_size)
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def perceive(self,state,action,reward,next_state,done):
        self.replay_buffer.append((state,action,reward,next_state,done))
        if len(self.replay_buffer) > self.hyperparameters.replay_size:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > self.hyperparameters.batch_size:
            self.train_Q_network()

    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            })[0]
        if random.random() <= self.epsilon:
            action = np.array([random.random() * 2 for _ in xrange(self.action_dim)])
        else:
            action = Q_value

        self.epsilon -= (self.hyperparameters.initial_epsilon - self.hyperparameters.final_epsilon)/self.max_evaluations
        return action

    def action(self,state):
        Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
        return Q_value


def test(agent, env):
    total_reward = 0

    state = env.reset()
    for _ in xrange(env.spec.timestep_limit):
        action = agent.action(state)
        state,reward,done,_ = env.step(action)
        total_reward += reward
        if done:
            break

    return total_reward

def train(agent, env):
    state = env.reset()
    for _ in xrange(env.spec.timestep_limit):
        action = agent.egreedy_action(state)
        next_state,reward,done,_ = env.step(action)
        agent.perceive(state,action,reward,next_state,done)
        state = next_state
        if done:
            break

def runExperiment(env_name, dataset, architecture, network_size, seed, max_evaluations):
    params = DqnParams()
    params.gamma = 0.9
    params.initial_epsilon = 0.9
    params.final_epsilon = 0.01
    params.replay_size = 10000
    params.batch_size = 32
    NUM_TEST_RUNS = 100

    np.random.seed(seed)
    file_name = "dqn_%s_%d_%s_%03d.dat" % (architecture, network_size, dataset, seed)
    f = open(file_name, 'w')
    f.write("seed\tevaluations\trun\tresult\n")

    env = gym.make(env_name)
    if dataset == "pendulum":
        agent = DQN_continous(env, network_size, params, max_evaluations)
    else:
        agent = DQN(env, network_size, params, max_evaluations)

    # Train for max_evaluations episodes
    for train_i in xrange(max_evaluations):
        if train_i % 100 == 0:
            for test_i in xrange(NUM_TEST_RUNS):
                result = test(agent, env)
                f.write("%03d\t%d\t%d\t%.3f\n" % (seed, train_i, test_i, result))

        train(agent, env)

    # Test for 100 episodes
    for test_i in xrange(NUM_TEST_RUNS):
        result = test(agent, env)
        f.write("%d\t%d\t%d\t%.3f\n" % (seed, max_evaluations, test_i, result))

    f.close()

if __name__ == '__main__':
    main()