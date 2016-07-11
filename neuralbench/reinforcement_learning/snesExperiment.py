import numpy as np

import gym

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import ReluLayer
from pybrain.structure.modules import LinearLayer
from pybrain.structure.modules import LSTMLayer
from pybrain.optimization import SNES

import neuralbench.algorithms.snes.core as snes_core

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


def run_network(nn, env, episode_count=1):
    discrete_output = isinstance(env.action_space, gym.spaces.discrete.Discrete)

    cumulated_reward = 0
    episode_count = 1

    for _ in xrange(episode_count):
        ob = env.reset()

        for _ in xrange(env.spec.timestep_limit):
            result = nn.activate(np.reshape(ob, -1))

            if discrete_output:
                action = np.argmax(result)
            else:
                assert(len(result) % 2 == 0)
                action = np.array([np.random.normal(result[i], abs(result[i+1]))
                                    if result[i+1] != 0 else result[i] for i in xrange(0, len(result), 2)])

            ob, reward, done, _ = env.step(action)
            cumulated_reward += reward

            if done:
                break

        nn.reset()

    return cumulated_reward

def configure_train_test(env_name, seed, build_network):
    def train_network(params):
        nn = build_network()
        nn._setParameters(np.array(params))

        env = gym.make(env_name)
        env.seed(snes_core.generation)

        result = run_network(nn, env)
        env.close()

        return result

    def test_network(params):
        # print params
        nn = build_network()
        nn._setParameters(np.array(params))

        env = gym.make(env_name)
        env.seed(seed)

        results = [run_network(nn, env) for _ in xrange(100)]
        env.close()
        return results

    return train_network, test_network

def evolve(env_name, seed, build_network, evaluations_per_generation_batch, max_batches):
    train_network, test_network = configure_train_test(env_name, seed, build_network)
    n = build_network()
    start_params = n.params

    run_snes, population_size = snes_core.configure_snes(train_network, start_params, minimize=False)
    iterator = run_snes()

    generations_per_batch = max(evaluations_per_generation_batch / population_size, 1)

    current_best = None
    i = 0

    while i < max_batches * generations_per_batch:
        for _ in xrange(generations_per_batch):
            current_best = iterator.next()
            i += 1
            assert i == snes_core.generation

        results = test_network(current_best)
        yield i * population_size, results


def runExperiment(env_name, dataset, architecture, network_size, seed, max_evaluations):
    np.random.seed(seed)
    file_name = "snes_%s_%d_%s_%03d.dat" % (architecture, network_size, dataset, seed)
    f = open(file_name, 'w')
    f.write("seed\tevaluations\trun\tresult\n")

    num_batches = 100
    evaluations_per_generation_batch = max_evaluations / num_batches

    build_network = net_configuration(architecture, network_size, env_name)

    evolution_iterator = evolve(env_name, seed, build_network, evaluations_per_generation_batch, num_batches)
    for evals, results in evolution_iterator:
        for test_i, result in enumerate(results):
            f.write("%03d\t%d\t%d\t%.3f\n" % (seed, evals, test_i, result))

        f.flush()

    f.close()


if __name__ == '__main__':
    import logging
    from tqdm import *
    gym.undo_logger_setup()
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    datasets = ["CartPole-v0", "Acrobot-v0", "MountainCar-v0", "Pendulum-v0"]
    evaluations_per_generation_batch = 5000

    print "Starting parameter sweep for snes"

    for seed in trange(1000):
        for run in xrange(5):
            total_reward = 0
            for env_name in datasets:
                build_network = net_configuration("simple", 10, env_name)
                evolution_iterator = evolve(env_name, seed, build_network, evaluations_per_generation_batch, 5)
                for generation, results in evolution_iterator:
                    result = sum(results)
                    if env_name == "Pendulum-v0":
                        result /= 10
                    total_reward += result

            print seed, run, total_reward


