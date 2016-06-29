import numpy as np

import gym

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import ReluLayer
from pybrain.structure.modules import LinearLayer
from pybrain.structure.modules import LSTMLayer
from pybrain.optimization import SNES

from neuralbench.classification.leea.leea_params import LeeaParams

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

class Gene(object):
    def __init__(self, chromosome, parent_fitness, evaluate):
        self.chromosome = chromosome
        self._evaluate = evaluate
        self._parent_fitness = parent_fitness
        self._fitness = None

    def __repr__(self):
        return "Fitness: %.4f - Parent Fitness: %.4f - Weight sum: %.4f\n" % (self._fitness, self._parent_fitness, np.sum(self.chromosome))

    def __str__(self):
        return "Fitness: %.4f - Parent Fitness: %.4f - Weight sum: %.4f\n" % (self._fitness, self._parent_fitness, np.sum(self.chromosome))

    @property
    def fitness(self):
        return self._fitness

    def calc_fitness(self, env, params):
        reward = self.evaluate(env)
        self._fitness = (1 - params.parent_fitness_decay) * self._parent_fitness + reward

        return reward

    def evaluate(self, env):
        return self._evaluate(self.chromosome, env)

    def mutate(self, params):
        child = []
        for parent_part in self.chromosome:
            mutation_indices = np.random.uniform(0, 1, parent_part.shape) >= params.mutation_rate
            mutations = np.random.uniform(-params.mutation_power, params.mutation_power, parent_part.shape)
            mutations[mutation_indices] = 0
            child.append(parent_part + mutations)
        return Gene(np.array(child), self.fitness, self._evaluate)

    def mate(self, partner):
        parent_fitness = (self.fitness + partner.fitness) / 2.0
        child1, child2 = self._uniform_crossover(partner)
        new_gene1 = Gene(child1, parent_fitness, self._evaluate)
        new_gene2 = Gene(child2, parent_fitness, self._evaluate)
        return new_gene1, new_gene2

    def _uniform_crossover(self, partner):
        child1 = []
        child2 = []
        assert(self.chromosome.shape == partner.chromosome.shape)
        idx = np.random.uniform(0, 1, self.chromosome.shape) < 0.5
        c1 = np.array(self.chromosome, copy=True)
        c2 = np.array(partner.chromosome, copy=True)
        c1[idx] = self.chromosome[idx]
        c2[idx] = partner.chromosome[idx]

        return c1, c2

def initialize_population(build_network, evaluation_function, params):
    return [Gene(build_network().params, 0, evaluation_function) for _ in xrange(params.population_size)]

def sort_population(pop):
    sorted_pop = sorted(pop, key=lambda individual: individual.fitness, reverse=True)
    return sorted_pop

def create_new_population(params, old_pop):
    new_pop = []
    while len(new_pop) < params.population_size:
        # Calculate probabilty for each individual
        fitness_scores = np.array([individual.fitness for individual in old_pop])
        probs = fitness_scores / sum(fitness_scores)

        if (np.random.uniform(0, 1) < params.sexual_reproduction_proportion):
            parents = np.random.choice(len(old_pop), 2, replace=False, p=probs)
            child1, child2 = old_pop[parents[0]].mate(old_pop[parents[1]])
            new_pop.append(child1)
            new_pop.append(child2)
        else:
            parent = np.random.choice(len(old_pop), 1, replace=False, p=probs)
            child = old_pop[parent[0]].mutate(params)
            new_pop.append(child)

    params.mutation_power = params.mutation_power * params.mutation_power_decay

    # Cut offspring besides population_size
    return new_pop[:params.population_size]


def train_network(env_name, seed, step_limit, max_evaluations, f, build_network, params):
    env = gym.make(env_name)
    discrete_output = isinstance(env.action_space, gym.spaces.discrete.Discrete)

    def run_network(chromosome, env=None):
        nn = build_network()
        nn._setParameters(np.array(chromosome)) 

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
                    try:
                        action = np.array([np.random.normal(result[i], abs(result[i+1])) if result[i+1] != 0 else result[i] for i in xrange(0, len(result), 2)])
                    except:
                        print "Something went wrong with: ", result
                        action = np.random.uniform(0, 1)

                ob, reward, done, _ = env.step(action)
                cum_reward += reward
                if done:
                    break

            nn.reset()

        return cum_reward

    # Build initial population
    population = initialize_population(build_network, run_network, params)

    num_generations = max_evaluations/params.population_size + 1
    elite_size = int(params.population_size * params.selection_proportion)

    best = None
    for generation in xrange(num_generations):
        for individual in population:
            env = gym.make(env_name)
            env.seed(generation)
            individual.calc_fitness(env, params)

        population = sort_population(population)
        best = population[0]
        population = create_new_population(params, population[:elite_size])
    

    env = gym.make(env_name)

    for run in xrange(100):
        cum_reward = best.evaluate(env)
        f.write("%03d\t%d\t%d\t%.3f\n" % (seed, (num_generations * params.population_size), run, cum_reward))
        if run % 10 == 0:
            f.flush()


def runExperiment(env_name, dataset, architecture, network_size, seed, step_limit, max_evaluations):
    params = LeeaParams()
    params.parent_fitness_decay = 0.05
    params.mutation_power_decay = 0.99
    params.sexual_reproduction_proportion = 0.5
    params.population_size = 50
    params.starting_mutation_power = 0.3
    params.mutation_power = params.starting_mutation_power
    params.mutation_rate = 0.4
    params.selection_proportion = 0.4

    np.random.seed(seed)
    file_name = "leea_%s_%d_%s_%03d.dat" % (architecture, network_size, dataset, seed)
    f = open(file_name, 'w')
    f.write("seed\tevaluations\trun\tresult\n")

    # TODO run experiment
    build_network = net_configuration(architecture, network_size, env_name)
    train_network(env_name, seed, step_limit, max_evaluations, f, build_network, params)

    f.close()

if __name__ == '__main__':
    import sys
    seed = int(sys.argv[1])

    datasets = ["CartPole-v0", "Acrobot-v0", "MountainCar-v0", "Pendulum-v0"]
    params = LeeaParams()
    params.random_initialization(seed = seed)
    max_evaluations = 5000

    for architecture in ["simple", "recurrent"]:
        for env_name in datasets:
            for network_size in [1, 5, 10, 40, 100, 300]:
                step_limit = gym.envs.registry.spec(env_name).timestep_limit
                dataset_name = env_name.split("-")[0].lower()
                file_identifier = "params_leea_%s_%d_%s_%03d-%s" % (architecture, network_size, dataset_name, seed, str(params).replace("\t", "_"))
                print "Starting parameter sweep for leea %s" % file_identifier
                file_name = "%s.dat" % (file_identifier)
                f = open(file_name, 'w')
                f.write("seed\tevaluations\trun\tresult\n")
                build_network = net_configuration(architecture, network_size, env_name)
                train_network(env_name, seed, step_limit, max_evaluations, f, build_network, params)

                f.close()