import numpy as np

import gym

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import ReluLayer
from pybrain.structure.modules import LinearLayer
from pybrain.structure.modules import LSTMLayer
from pybrain.optimization import SNES

from neuralbench.classification.cosyne.cosyne_params import CosyneParams

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

    env.close()

    if architecture == "simple":
        return build_simple_network(input_size, network_size, output_size)
    elif architecture == "recurrent":
        return build_recurrent_network(input_size, network_size, output_size)
    else:
        raise NameError('No Architecture with that name')

def initialize_population(build_network, params):
    return np.array([build_network().params for _ in xrange(params.population_size)])

def uniform_crossover(parent1, parent2):
    assert(parent1.shape == parent2.shape)
    idx = np.random.uniform(0, 1, parent1.shape) < 0.5
    child1 = parent1
    child2 = parent2
    child1[idx] = parent2[idx]
    child2[idx] = parent1[idx]

    return child1, child2

def uniform_mutation(parent, params):
    mutation_indices = np.random.uniform(0, 1, parent.shape) >= params.mutation_rate
    mutations = np.random.uniform(-params.mutation_power, params.mutation_power, parent.shape)
    mutations[mutation_indices] = 0
    return parent + mutations

# Shuffles weights in one subpoplation / shuffles columns
def permute_inplace(population):
    transposed = population.T
    map(np.random.shuffle, transposed)

def create_new_generation(old_pop, results, params):
    top_idx = int(len(old_pop) * params.selection_proportion)
    top_idx = 2 if top_idx < 2 else top_idx
    elite = np.array(old_pop[0:top_idx], copy=True)
    offspring = []
    needed_offsprings = len(old_pop) - top_idx
    for _ in xrange(needed_offsprings/2 + 1):
        # parents = np.random.choice(len(elite), 2, replace=False, p=softmax(-np.array(results[0:top_idx])))
        # Substract max, so that lowest will have highest prob
        # new_results = np.array(results[0:top_idx]) - (results[-1] + 1)
        fitness_scores = results[0:top_idx]
        fitness_scores = fitness_scores - np.min(fitness_scores) + 1
        probs = fitness_scores / float(sum(fitness_scores))
        parents = np.random.choice(len(elite), 2, replace=False, p=probs)
        child1, child2 = uniform_crossover(elite[parents[0]], elite[parents[1]])
        child1 = uniform_mutation(child1, params)
        child2 = uniform_mutation(child2, params)
        offspring.append(child1)
        offspring.append(child2)

    permute_inplace(elite)
    new_pop = np.append(elite, offspring, axis=0)
    new_pop = new_pop[:len(old_pop)]
    assert(old_pop.shape == new_pop.shape)

    return new_pop


def train_network(env_name, seed, step_limit, max_evaluations, f, build_network, params):
    env = gym.make(env_name)
    discrete_output = isinstance(env.action_space, gym.spaces.discrete.Discrete)
    env.close()

    def run_network(chromosome, current_seed=None, input_env=None):
        nn = build_network()
        nn._setParameters(np.array(chromosome)) 

        if input_env == None:
            current_env = gym.make(env_name)
        else:
            current_env = input_env
        if current_seed != None:
            current_env.seed(current_seed)

        cum_reward = 0
        episode_count = 1

        for _ in xrange(episode_count):
            ob = current_env.reset()

            for _ in xrange(step_limit):
                result = nn.activate(np.reshape(ob, -1))
                
                if discrete_output:
                    action = np.argmax(result)
                else:
                    assert(len(result) % 2 == 0)
                    try:
                        action = np.array([np.random.normal(result[i], abs(result[i+1])) if result[i+1] != 0 else result[i] for i in xrange(0, len(result), 2)])
                    except:
                        print "Something went wrong with: ", result
                        action = np.random.uniform(0, 1)

                ob, reward, done, _ = current_env.step(action)
                cum_reward += reward
                if done:
                    break

            nn.reset()

        if input_env == None:
            current_env.close()

        return cum_reward

    # Build initial population
    population = initialize_population(build_network, params)

    num_generations = max_evaluations/params.population_size + 1

    best = None
    for generation in xrange(num_generations):
        results = np.array([run_network(individium, current_seed=generation) for individium in population])
        sort_idx = np.argsort(results)[::-1]
        sorted_pop = population[sort_idx]
        sorted_results = results[sort_idx]
        best = sorted_pop[0]
        population = create_new_generation(sorted_pop, sorted_results, params)
        hp = hpy()
        before = hp.heap()
        print before
        print "generation: %d, best: %d" % (generation, sorted_results[0])

    env = gym.make(env_name)
    net = build_network()
    net._setParameters(best) 
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

        f.write("%03d\t%d\t%d\t%.3f\n" % (seed, (num_generations * params.population_size), run, cum_reward))
        if run % 10 == 0:
            f.flush()

    env.close()


def runExperiment(env_name, dataset, architecture, network_size, seed, step_limit, max_evaluations):
    params = CosyneParams()
    params.population_size = 40
    params.mutation_power = 0.4
    params.mutation_rate = 0.4
    params.selection_proportion = 0.2

    np.random.seed(seed)
    file_name = "cosyne_%s_%d_%s_%03d.dat" % (architecture, network_size, dataset, seed)
    f = open(file_name, 'w')
    f.write("seed\tevaluations\trun\tresult\n")

    # TODO run experiment
    build_network = net_configuration(architecture, network_size, env_name)
    train_network(env_name, seed, step_limit, max_evaluations, f, build_network, params)

    f.close()

if __name__ == '__main__':
    import logging
    gym.undo_logger_setup()
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    import sys
    seed = int(sys.argv[1])

    datasets = ["CartPole-v0", "Acrobot-v0", "MountainCar-v0", "Pendulum-v0"]
    params = CosyneParams()
    params.random_initialization(seed = seed)
    max_evaluations = 5000

    for architecture in ["simple", "recurrent"]:
        for env_name in datasets:
            for network_size in [1, 5, 10, 40, 100, 300]:
                step_limit = gym.envs.registry.spec(env_name).timestep_limit
                dataset_name = env_name.split("-")[0].lower()
                file_identifier = "params_cosyne_%s_%d_%s_%03d-%s" % (architecture, network_size, dataset_name, seed, str(params).replace("\t", "_"))
                print "Starting parameter sweep for cosyne %s" % file_identifier
                file_name = "%s.dat" % (file_identifier)
                f = open(file_name, 'w')
                f.write("seed\tevaluations\trun\tresult\n")
                build_network = net_configuration(architecture, network_size, env_name)
                train_network(env_name, seed, step_limit, max_evaluations, f, build_network, params)

                f.close()

