import pickle
import numpy as np
import gym

import MultiNEAT as NEAT

from neuralbench.algorithms.neat.params import standard_initialization, random_initialization, write_params_header, write_params
from neuralbench.algorithms.neat.core import configure_neat


def run_network(genome, env, episode_count=1):
    discrete_output = isinstance(env.action_space, gym.spaces.discrete.Discrete)
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    cumulated_reward = 0

    episode_count = 1
    for _ in xrange(episode_count):
        ob = env.reset()
        net.Flush()

        for _ in xrange(env.spec.timestep_limit):
            net.Input(np.reshape(ob, -1))
            for _ in xrange(1):
                net.Activate()
            o = net.Output()
            if discrete_output:
                action = np.argmax(o)
            else:
                assert(len(o) % 2 == 0)
                action = np.array([np.random.normal(o[i], abs(o[i+1])) for i in xrange(0, len(o), 2)])
            ob, reward, done, _ = env.step(action)
            cumulated_reward += reward

            if done:
                break

    return cumulated_reward

def train_network(genome, generation=0):
    env = gym.make(env_name)
    env.seed(generation)
    result = run_network(genome, env)
    env.close()
    return result

def test_network(genome):
    env = gym.make(env_name)
    results = [run_network(genome, env) for _ in xrange(100)]
    env.close()
    return results

def evolve(env_name, seed, params, num_generations, max_generations):
    env = gym.make(env_name)
    discrete_output = isinstance(env.action_space, gym.spaces.discrete.Discrete)

    if discrete_output:
        g = NEAT.Genome(0, len(np.reshape(env.observation_space.sample(), -1)), 0, env.action_space.n, False, 
                        NEAT.ActivationFunction.LINEAR, NEAT.ActivationFunction.RELU, 0, params)
    else:
        g = NEAT.Genome(0, len(np.reshape(env.observation_space.sample(), -1)), 0, 2*len(np.reshape(env.action_space.sample(), -1)), False, 
                NEAT.ActivationFunction.LINEAR, NEAT.ActivationFunction.RELU, 0, params)

    env.close()
    population = NEAT.Population(g, params, True, 1.0, seed)
    population.RNG.Seed(seed)

    run_neat = configure_neat(population, train_network)
    iterator = run_neat()

    current_best = None
    i = 0
    while i < max_generations:
        for _ in xrange(num_generations):
            generation, current_best = iterator.next()
            i += 1

        best = pickle.loads(current_best)
        results = test_network(best)
        yield i, results


def runExperiment(env_name, dataset, seed, step_limit, max_evaluations):
    np.random.seed(seed)
    file_name = "neat_neat_%s_%03d.dat" % (dataset, seed)
    f = open(file_name, 'w')
    f.write("seed\tevaluations\trun\tresult\n")

    params = standard_initialization()
    params.RecurrentProb = 0.4
    NUM_TESTS = 100
    num_generations = (max_evaluations/NUM_TESTS)/params.PopulationSize + 1

    evolution_iterator = evolve(env_name, seed, params, num_generations, NUM_TESTS*num_generations)
    for generation, results in evolution_iterator:
        for test_i, result in results:
            f.write("%03d\t%d\t%d\t%.3f\n" % (seed, (generation * params.PopulationSize), test_i, result))

    f.close()

if __name__ == '__main__':
    import logging
    from tqdm import *
    gym.undo_logger_setup()
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    datasets = ["CartPole-v0", "Acrobot-v0", "MountainCar-v0", "Pendulum-v0"]
    max_evaluations = 5000

    print "Starting parameter sweep for neat"
    file_identifier = "params_neat"
    file_name = "%s.dat" % (file_identifier)
    f = open(file_name, 'w')
    f.write("result\tevaluations\tseed\t%s\n" % (write_params_header()))

    for seed in trange(1000):
        params = random_initialization(seed = seed)
        num_generations = max_evaluations/params.PopulationSize + 1
        total_reward = 0
        for env_name in datasets:
            evolution_iterator = evolve(env_name, seed, params, num_generations, num_generations)
            for generation, results in evolution_iterator:
                result = sum(results)
                if env_name == "Pendulum-v0":
                    result /= 10
                total_reward += result

        f.write("%0.3f\t%d\t%d\t%s\n" % (-total_reward, num_generations * params.PopulationSize, seed, write_params(params)))
        f.flush()

    f.close()

