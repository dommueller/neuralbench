import numpy as np

import gym

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import ReluLayer
from pybrain.structure.modules import LinearLayer
from pybrain.structure.modules import LSTMLayer
from pybrain.optimization import SNES

from neuralbench.algorithms.leea.leea_params import LeeaParams
import neuralbench.algorithms.leea.core as leea_core

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

def configure_train_test(env_name, build_network):
    def train_network(individual, seed):
        nn = build_network()
        nn._setParameters(np.array(individual))

        env = gym.make(env_name)
        env.seed(seed)

        result = run_network(nn, env)
        env.close()

        return result

    def test_network(individual, seed):
        nn = build_network()
        nn._setParameters(np.array(individual))

        env = gym.make(env_name)
        env.seed(seed)

        results = [run_network(nn, env) for _ in xrange(100)]
        env.close()
        return results

    return train_network, test_network

def evolve(env_name, seed, params, build_network, max_evaluations, num_batches):
    train_network, test_network = configure_train_test(env_name, build_network)

    population = leea_core.initialize_population(build_network, train_network, test_network, params)

    run_leea = leea_core.configure_leea(population, train_network, params)
    iterator = run_leea()

    generations_per_batch = max((max_evaluations / num_batches) / params.population_size, 1)

    current_best = None
    i = 0

    while i * params.population_size < max_evaluations:
        for _ in xrange(generations_per_batch):
            generation, current_best = iterator.next()
            i += 1
            assert i == generation

        results = current_best.test(seed)
        yield i * params.population_size, results

def runExperiment(env_name, dataset, architecture, network_size, seed, max_evaluations, num_batches):
    params = LeeaParams()
    # params.parent_fitness_decay = 0.05
    # params.mutation_power_decay = 0.99
    # params.sexual_reproduction_proportion = 0.5
    # params.population_size = 50
    # params.starting_mutation_power = 0.3
    # params.mutation_power = params.starting_mutation_power
    # params.mutation_rate = 0.4
    # params.selection_proportion = 0.4

    params.parent_fitness_decay = 0.85
    params.mutation_power_decay = 0.99
    params.sexual_reproduction_proportion = 0.5
    params.population_size = 80
    params.starting_mutation_power = 6.7
    params.mutation_power = params.starting_mutation_power
    params.mutation_rate = 0.92
    params.selection_proportion = 0.75

    np.random.seed(seed)
    file_name = "leea_%s_%d_%s_%03d.dat" % (architecture, network_size, dataset, seed)
    f = open(file_name, 'w')
    f.write("seed\tevaluations\trun\tresult\n")

    build_network = net_configuration(architecture, network_size, env_name)

    evolution_iterator = evolve(env_name, seed, params, build_network, max_evaluations, num_batches)
    for evals, results in evolution_iterator:
        for test_i, result in enumerate(results):
            f.write("%03d\t%d\t%d\t%.3f\n" % (seed, evals, test_i, result))

        f.flush()

    f.close()

if __name__ == '__main__':
    import logging
    gym.undo_logger_setup()
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    import sys
    seed = int(sys.argv[1])

    datasets = ["CartPole-v0", "Acrobot-v0", "MountainCar-v0", "Pendulum-v0"]
    params = LeeaParams()
    params.random_initialization(seed = seed)
    max_evaluations = 5000

    print "Starting parameter sweep for leea"

    for seed in trange(1000):
        for run in xrange(5):
            total_reward = 0
            for env_name in datasets:
                build_network = net_configuration("simple", 10, env_name)
                evolution_iterator = evolve(env_name, seed, params, build_network, max_evaluations, 1)
                for generation, results in evolution_iterator:
                    result = sum(results)
                    if env_name == "Pendulum-v0":
                        result /= 10
                    total_reward += result

            print seed, run, total_reward