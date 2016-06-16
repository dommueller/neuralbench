import copy
import numpy as np
import tensorflow as tf

import leeaNetworks
from leea_params import LeeaParams
from neuralbench.classification.dataset.create import dense_to_one_hot

class Gene(object):
    def __init__(self, chromosome, parent_fitness, evaluate):
        self.chromosome = chromosome
        self._evaluate = evaluate
        self._parent_fitness = parent_fitness
        self._fitness = None

    @property
    def fitness(self):
        return self._fitness

    def calc_fitness(self, params, X, y, sess):
        acc, c = self.evaluate(X, y, sess)
        if c > 10000:
            print "LOSS is over 10000", c

        # Need a fitness value to maximize
        own_fitness = 10000.0 - c
        self._fitness = (1 - params.parent_fitness_decay) * self._parent_fitness + own_fitness
        return acc, c

    def evaluate(self, X, y, sess):
        return self._evaluate(sess, self.chromosome, X, y)

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
        for parent1, parent2 in zip(self.chromosome, partner.chromosome):
            assert(parent1.shape == parent2.shape)
            idx = np.random.uniform(0, 1, parent1.shape) < 0.5
            c1 = copy.deepcopy(parent1[:])
            c2 = copy.deepcopy(parent2[:])
            c1[idx] = parent2[idx]
            c2[idx] = parent1[idx]
            child1.append(c1)
            child2.append(c2)

        return np.array(child1), np.array(child2)

def initialize_population(pop_size, chromosome_template, alpha, evaluation_function):
    return [Gene(np.array([np.random.uniform(-alpha, alpha, template_part.shape) 
                            for template_part in chromosome_template])
                        , 0, evaluation_function) for _ in xrange(pop_size)]

def sort_population(pop):
    sorted(pop, key=lambda individual: individual.fitness, reverse=True)

def create_new_population(params, old_pop):
    pop_size = len(old_pop)
    new_pop = []
    while len(new_pop) < pop_size:
        # Calculate probabilty for each individual
        fitness_scores = np.array([individual.fitness for individual in old_pop])
        probs = fitness_scores / sum(fitness_scores)

        if (np.random.uniform(0, 1) < params.sexual_reproduction_proportion):
            parents = np.random.choice(pop_size, 2, replace=False, p=probs)
            child1, child2 = old_pop[parents[0]].mate(old_pop[parents[1]])
            new_pop.append(child1)
            new_pop.append(child2)
        else:
            parent = np.random.choice(pop_size, 1, replace=False, p=probs)
            child = old_pop[parent[0]].mutate(params)
            new_pop.append(child)

    params.mutation_power = params.mutation_power * params.mutation_power_decay

    # Cut offspring besides population_size
    return new_pop[:pop_size]

def configure_for_training(params, max_evaluations, n_classes, eval_net, weights_template, seed, f):

    def train_network(X_train, y_train, X_validate, y_validate, X_test, y_test, test_split=0, validate_split=0):
        file_start = "%d\t%d\t%d" % (seed, test_split, validate_split)

        assert(n_classes == len(np.unique(y_train)))
        y_train_one_hot = dense_to_one_hot(y_train, n_classes)
        assert(n_classes == len(np.unique(y_validate)))
        y_validate = dense_to_one_hot(y_validate, n_classes)
        assert(n_classes == len(np.unique(y_test)))
        y_test = dense_to_one_hot(y_test, n_classes)

        pop = initialize_population(params.population_size, weights_template, params.initial_weight_range, eval_net)

        with tf.Session() as sess:
            validate_results = np.array([[acc, cost] for acc, cost in [individual.evaluate(X_validate, y_validate, sess) for individual in pop]])
            best_validation_acc_index = np.argmax(validate_results, axis=0)[0]
            acc, cost = pop[best_validation_acc_index].evaluate(X_test, y_test, sess)

            f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "val", 0, "acc", np.max(validate_results, axis=0)[0]))
            f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "test", 0, "acc", acc))
            f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "val", 0, "cost", np.min(validate_results, axis=0)[1]))
            f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "test", 0, "cost", cost))

            evaluations_per_generation = params.population_size * params.batch_size
            num_generations = max_evaluations/(evaluations_per_generation) + 1

            # Used to sample a batch with same class ratios
            from sklearn.cross_validation import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(y_train.reshape(-1), num_generations, train_size=params.batch_size, random_state=seed)
            
            for generation, (batch_index, _) in enumerate(sss):
                X_current = X_train[batch_index]
                y_current = y_train_one_hot[batch_index]

                train_results = np.array([[acc, cost] for acc, cost in [individual.calc_fitness(params, X_current, y_current, sess) for individual in pop]])

                sort_population(pop)
                pop = create_new_population(params, pop)

                validate_results = np.array([[acc, cost] for acc, cost in [individual.evaluate(X_validate, y_validate, sess) for individual in pop]])
                best_validation_acc_index = np.argmax(validate_results, axis=0)[0]
                acc, cost = pop[best_validation_acc_index].evaluate(X_test, y_test, sess)
                
                train_evaluations = (generation + 1) * evaluations_per_generation

                f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "train", train_evaluations, "acc", np.max(train_results, axis=0)[0]))
                f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "val", train_evaluations, "acc", np.max(validate_results, axis=0)[0]))
                f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "test", train_evaluations, "acc", acc))
                f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "train", train_evaluations, "cost", np.min(train_results, axis=0)[1]))
                f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "val", train_evaluations, "cost", np.min(validate_results, axis=0)[1]))
                f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "test", train_evaluations, "cost", cost))
                if generation % 100 == 0:
                    f.flush()

    return train_network

def runExperiment(architecture, dataset, seed, max_evaluations, num_samples):
    from neuralbench.classification.dataset.create import run_validate_splits
    params = LeeaParams()
    params.parent_fitness_decay = 0.2
    params.starting_mutation_power = 0.03
    params.mutation_power_decay = 0.99
    params.sexual_reproduction_proportion = 0.5
    params.population_size = 200
    params.mutation_power = 0.03
    params.mutation_rate = 0.04
    params.selection_proportion = 0.4
    params.initial_weight_range = 4.
    params.batch_size = num_samples
    np.random.seed(seed)

    eval_genotype, weights_template = leeaNetworks.createArchitecture(architecture, dataset["name"])

    np.random.seed(seed)
    file_name = "leea_%s_%s_%03d_e%010d_s%05d.dat" % (architecture, dataset["name"], seed, max_evaluations, num_samples)
    f = open(file_name, 'w')
    f.write("seed\ttest_split\tvalidation_split")
    f.write("\tpopulation_size\tmutation_power\tmutation_rate\tselection_proportion\tbatch_size\tinitial_weight_range")
    f.write("\tevaluation_data\tevaluations\tfitness_type\tresult\n")

    if dataset["name"] == "mnist":
        train_network = configure_for_training(params, max_evaluations, 10, eval_genotype, weights_template, seed, f)
        run_validate_splits(train_network, dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"], folds=10, seed=seed)
    else:
        train_network = configure_for_training(params, max_evaluations, 2, eval_genotype, weights_template, seed, f)
        run_validate_splits(train_network, dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"], folds=10, seed=seed)

    f.close()

if __name__ == '__main__':
    print "Starting parameter sweep for leea"
    import sys
    from tqdm import *
    from neuralbench.classification.dataset.create import createDataSet, run_test_validate_splits
    
    dataset_name = "spiral"
    architecture = "deep"

    # for dataset_name in tqdm(["spiral", "circle"]):
    #     for architecture in tqdm(["perceptron", "small", "big", "deep"]):
    
    X_train, y_train, X_test, y_test = createDataSet(dataset_name)
    eval_net, weights_template = leeaNetworks.createArchitecture(architecture, dataset_name)

    PARENT_FITNESS_DECAY = 0.2
    STARTING_MUTATION_POWER = 0.03
    MUTATION_POWER = 0.03
    MUTATION_RATE = 0.4

    # POPULATION_SIZE = 1000
    SELECTION_PROPORTION = 0.4
    MUTATION_POWER_DECAY = 0.99
    SEXUAL_REPRODUCTION_PROPORTION = 0.5
    
    INITIAL_WEIGHT_RANGE = 4
    seed = 0
    file_identifier = str(sys.argv[1]) + "/params_leea_%s_%s_%03d_%.2f_%.2f_%.2f_%.1f" % (architecture, dataset_name, seed,
                                MUTATION_POWER, MUTATION_RATE, SELECTION_PROPORTION, INITIAL_WEIGHT_RANGE)
    file_name = "%s.dat" % (file_identifier)
    f = open(file_name, 'w')
    f.write("seed\ttest_split\tvalidation_split")
    f.write("\tpopulation_size\tmutation_power\tmutation_rate\tselection_proportion\tbatch_size\tinitial_weight_range")
    f.write("\tevaluation_data\tevaluations\tfitness_type\tresult\n")

    for population_size in tqdm([10, 40, 100, 400, 1000]):
        for batch_size in tqdm([10, 20, 40, 80, 100, 150, 200, 250]):
            standard_params = "%d\t%0.2f\t%0.2f\t%0.1f\t%d\t%0.1f" % (population_size, MUTATION_POWER, MUTATION_RATE,
                    SELECTION_PROPORTION, batch_size, INITIAL_WEIGHT_RANGE)

            # configure_for_training(params, max_evaluations, n_classes, eval_genotype, num_network_weights, seed, file_identifier)
            train_network = configure_for_training(standard_params, 300000, 2, eval_net, weights_template, seed, f)
            run_test_validate_splits(train_network, X_train, y_train, folds=10)

    f.close()



