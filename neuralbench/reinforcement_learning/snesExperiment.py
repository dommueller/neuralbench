import numpy as np

import gym

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import ReluLayer
from pybrain.structure.modules import LinearLayer
from pybrain.structure.modules import LSTMLayer
from pybrain.optimization import SNES

def build_simple_network(input_size, network_size, output_size):
    return lambda: buildNetwork(input_size, network_size, output_size, hiddenclass=ReluLayer, outclass=LinearLayer, outputbias=True, recurrent=False)

def build_recurrent_network(input_size, network_size, output_size):
    return lambda: buildNetwork(input_size, network_size, output_size, hiddenclass=LSTMLayer, outclass=LinearLayer, outputbias=True, recurrent=True)

def net_configuration(architecture, network_size, env_name):
    env = gym.make(env_name)
    discrete_output = isinstance(env.action_space, gym.spaces.discrete.Discrete)
    if discrete_output:
        input_size = len(np.reshape(env.observation_space.sample(), -1))
        output_size = env.action_space.n
    else:
        input_size = len(np.reshape(env.observation_space.sample(), -1))
        output_size = 2 * len(np.reshape(env.action_space.sample(), -1))

    if architecture == "simple":
        return build_simple_network(input_size, network_size, output_size)
    elif architecture == "recurrent":
        return build_recurrent_network(input_size, network_size, output_size)
    else:
        raise NameError('No Architecture with that name')




def train_network(env_name, seed, step_limit, max_evaluations, f, build_network):
    env = gym.make(env_name)
    discrete_output = isinstance(env.action_space, gym.spaces.discrete.Discrete)
    
    def objF(params):
        nn = build_network()
        nn._setParameters(np.array(params)) 

        env = gym.make(env_name)
        env.seed(l.numLearningSteps)

        cum_reward = 0
        episode_count = 1

        for _ in xrange(episode_count):
            ob = env.reset()

            for _ in xrange(step_limit):
                result = nn.activate(np.reshape(ob, -1))
                
                if discrete_output:
                    action = np.argmax(result)
                else:
                    assert(len(result) % 2 == 0)
                    action = np.array([np.random.normal(result[i], abs(result[i+1])) for i in xrange(0, len(result), 2)])

                ob, reward, done, _ = env.step(action)
                cum_reward += reward
                if done:
                    break

            nn.reset()

        return cum_reward

    # Set enviroment

    # Build net for initial random params
    n = build_network()
    x0 = n.params

    l = SNES(objF, x0, verbose=False)
    l.minimize = False
    num_generations = max_evaluations/l.batchSize + 1
    l.maxEvaluations = num_generations * num_generations

    best = x0
    for generation in xrange(num_generations):
        result = l.learn(additionalLearningSteps=1)
        best = result[0]

    
    net = build_network()
    net._setParameters(np.array(best)) 
    for run in xrange(100):
        cum_reward = 0
        episode_count = 1

        for _ in xrange(episode_count):
            ob = env.reset()

            for _ in xrange(step_limit):
                result = net.activate(np.reshape(ob, -1))

                if discrete_output:
                    action = np.argmax(result)
                else:
                    assert(len(result) % 2 == 0)
                    action = np.array([np.random.normal(result[i], abs(result[i+1])) for i in xrange(0, len(result), 2)])

                ob, reward, done, _ = env.step(action)
                cum_reward += reward
                if done:
                    break

            net.reset()

        f.write("%03d\t%d\t%d\t%.3f\n" % (seed, (num_generations * l.batchSize), run, cum_reward)) 


def runExperiment(env_name, dataset, architecture, network_size, seed, step_limit, max_evaluations):
    np.random.seed(seed)
    file_name = "snes_%s_%d_%s_%03d.dat" % (architecture, network_size, dataset, seed)
    f = open(file_name, 'w')
    f.write("seed\tevaluations\trun\tresult\n")

    # TODO run experiment
    build_network = net_configuration(architecture, network_size, env_name)
    train_network(env_name, seed, step_limit, max_evaluations, f, build_network)

    f.close()