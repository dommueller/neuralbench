import numpy as np
import tensorflow as tf

import cosyneNetworks
from cosyne_params import CosyneParams


# Taken from 
# https://github.com/tensorflow/tensorflow/blob/1d76583411038767f673a0c96174c80eaf9ff42f/tensorflow/g3doc/tutorials/mnist/input_data.py
def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  labels_dense = np.array(labels_dense, dtype="int32")
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Population layout:
# n: number of subpopulations (each is a weight for a connection in the network)
# m: size of subpopulations / weights per subpopulation
# m x n matrix
# each row is possible phenotype
# each column contains all weights of one subpopulation 
def initialize_population(n, m, alpha):
    return np.random.uniform(-alpha, alpha, [m, n])

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

def createNewPop(old_pop, results, params):
    top_idx = int(len(old_pop) * params.selection_proportion)
    top_idx = 2 if top_idx < 2 else top_idx
    elite = old_pop[0:top_idx]
    offspring = []
    needed_offsprings = len(old_pop) - top_idx
    for _ in xrange(needed_offsprings/2 + 1):
        # parents = np.random.choice(len(elite), 2, replace=False, p=softmax(-np.array(results[0:top_idx])))
        # Substract max, so that lowest will have highest prob
        # new_results = np.array(results[0:top_idx]) - (results[-1] + 1)
        new_results = 1.0 / (results[0:top_idx] + 1)
        probs = new_results / sum(new_results)
        parents = np.random.choice(len(elite), 2, replace=False, p=probs)
        child1, child2 = uniform_crossover(elite[parents[0]], elite[parents[1]])
        child1 = uniform_mutation(child1, params)
        child2 = uniform_mutation(child2, params)
        offspring.append(child1)
        offspring.append(child2)

    permute_inplace(elite)
    new_pop = old_pop[:]
    new_pop[0:top_idx] = elite
    new_pop[top_idx:] = offspring[0:needed_offsprings]

    assert(len(old_pop) == len(new_pop))

    return new_pop

def trainNetwork(data, n_classes, buildNet, num_network_weights, file, seed, max_evaluations, params):
    X_train = data["X_train"]
    y_train = dense_to_one_hot(data["y_train"], n_classes)
    X_test = data["X_test"]
    y_test = dense_to_one_hot(data["y_test"], n_classes)

    population_size = 1000
    pop = initialize_population(num_network_weights, population_size, 1)

    eval_genotype = buildNet(X_train.shape[1], y_train.shape[1])
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        for i in xrange((max_evaluations/population_size)):
        # for i in xrange(1):
            sampled_data = np.random.choice(len(X_train), params.batch_size, replace=False)
            cur_data = X_train[sampled_data]
            cur_label = y_train[sampled_data]

            results = np.array([cost for acc, cost in [eval_genotype(sess, chromosome, cur_data, cur_label) for chromosome in pop]])
            # print "Results", results
            # print "Min", np.argmin(results)
            acc, _ = eval_genotype(sess, pop[np.argmin(results)], X_test, y_test)
            evals = (i + 1) * params.batch_size * population_size
            file.write("cosyne %d %d %f\n" % (seed, evals, acc))
            # Sort population by cost
            sort_idx = np.argsort(results)
            sorted_pop = pop[sort_idx]
            sorted_results = results[sort_idx]

            pop = createNewPop(sorted_pop, sorted_results, params)
            if i % 100 == 0:
                file.flush()


def runExperiment(architecture, dataset, seed, max_evaluations, num_samples):
    params = CosyneParams()
    params.population_size = 40
    params.mutation_power = 0.3
    params.mutation_rate = 0.4
    params.selection_proportion = 0.4
    params.initial_weight_range = 4.
    params.batch_size = num_samples
    np.random.seed(seed)
    file_name = "cosyne_%s_%s_%03d_e%010d_s%05d.dat" % (architecture, dataset["name"], seed, max_evaluations, num_samples)
    f = open(file_name, 'w')
    f.write("seed\ttest_split\tvalidation_split")
    f.write("\tpopulation_size\tmutation_power\tmutation_rate\tselection_proportion\tbatch_size\tinitial_weight_range")
    f.write("\tevaluation_data\tevaluations\tfitness_type\tresult\n")
    buildNet, num_network_weights = cosyneNetworks.createArchitecture(architecture, dataset["name"])
    if dataset["name"] == "mnist":
        from neuralbench.classification.dataset.create import run_validate_splits
        eval_genotype = buildNet(28*28, 10)
        train_network = configure_for_training(params, max_evaluations, 10, eval_genotype, num_network_weights, seed, f)
        run_validate_splits(train_network, dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"], folds=10, seed=seed)
    else:
        from neuralbench.classification.dataset.create import run_validate_splits
        eval_genotype = buildNet(2, 2)
        train_network = configure_for_training(params, max_evaluations, 2, eval_genotype, num_network_weights, seed, f)
        run_validate_splits(train_network, dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"], folds=10, seed=seed)

    f.close()


def configure_for_training(params, max_evaluations, n_classes, eval_genotype, num_network_weights, seed, f):
    def train_network(X_train, y_train, X_validate, y_validate, X_test, y_test, test_split=0, validate_split=0):
        file_start = "%d\t%d\t%d" % (seed, test_split, validate_split)

        assert(n_classes == len(np.unique(y_train)))
        y_train_one_hot = dense_to_one_hot(y_train, n_classes)
        assert(n_classes == len(np.unique(y_validate)))
        y_validate = dense_to_one_hot(y_validate, n_classes)
        assert(n_classes == len(np.unique(y_test)))
        y_test = dense_to_one_hot(y_test, n_classes)

        pop = initialize_population(num_network_weights, params.population_size, params.initial_weight_range)

        with tf.Session() as sess:
            # validate_results = np.array([[acc, cost] for acc, cost in [eval_genotype(sess, chromosome, X_validate, y_validate) for chromosome in pop]])
            # best_validation_acc_index = np.argmax(validate_results, axis=0)[0]
            # acc, cost = eval_genotype(sess, pop[best_validation_acc_index], X_test, y_test)

            # f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, str(params), "val", 0, "acc", np.max(validate_results, axis=0)[0]))
            # f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "test", 0, "acc", acc))
            # f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, str(params), "val", 0, "cost", np.min(validate_results, axis=0)[1]))
            # f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "test", 0, "cost", cost))

            evaluations_per_generation = params.population_size * params.batch_size
            num_generations = max_evaluations/(evaluations_per_generation) + 1
            # Used to sample a batch with same class ratios
            from sklearn.cross_validation import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(y_train.reshape(-1), num_generations, train_size=params.batch_size, random_state=seed)
            
            generation = 0
            for batch_index, _ in sss:
                generation += 1
                X_current = X_train[batch_index]
                y_current_correct = y_train_one_hot[batch_index]
                disturbation_indices = np.random.uniform(0, 1, y_current_correct.shape) >= 0.5
                y_current_disturbed = np.random.randint(n_classes, size=y_current_correct.shape)
                y_current_disturbed[disturbation_indices] = y_current_correct[disturbation_indices]

                train_results = np.array([[acc, cost] for acc, cost in [eval_genotype(sess, chromosome, X_current, y_current_correct) for chromosome in pop]])
                
                train_cost = train_results[:,1]
                sort_idx = np.argsort(train_cost)
                sorted_pop = pop[sort_idx]
                sorted_results = train_cost[sort_idx]

                pop = createNewPop(sorted_pop, sorted_results, params)

                validate_results = np.array([[acc, cost] for acc, cost in [eval_genotype(sess, chromosome, X_validate, y_validate) for chromosome in pop]])
                best_validation_acc_index = np.argmax(validate_results, axis=0)[0]
                acc, cost = eval_genotype(sess, pop[best_validation_acc_index], X_test, y_test)

                train_evaluations = (generation) * evaluations_per_generation

                f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "train", train_evaluations, "acc", np.max(train_results, axis=0)[0]))
                f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "val", train_evaluations, "acc", np.max(validate_results, axis=0)[0]))
                f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "test", train_evaluations, "acc", acc))
                f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "train", train_evaluations, "cost", np.min(train_results, axis=0)[1]))
                f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "val", train_evaluations, "cost", np.min(validate_results, axis=0)[1]))
                f.write("%s\t%s\t%s\t%d\t%s\t%f\n" % (file_start, params, "test", train_evaluations, "cost", cost))
                # if generation % 100 == 0:
                f.flush()

    return train_network

if __name__ == '__main__':
    import sys
    import random
    from tqdm import *
    from neuralbench.classification.dataset.create import createDataSet, run_test_validate_splits
    seed = int(sys.argv[1])
    
    dataset_name = "mnist"
    # architecture = random.choice(["perceptron", "small", "big", "deep"])
    params = CosyneParams()
    params.random_initialization(seed = seed)
    max_evaluations = 100000
    while (params.population_size * params.batch_size > max_evaluations/2):
        params.batch_size = random.randint(10, 200)

    X_train, y_train, X_test, y_test = createDataSet(dataset_name)
    for architecture in ["perceptron", "small", "big", "deep"]:
        buildNet, num_network_weights = cosyneNetworks.createArchitecture(architecture, dataset_name)
        eval_genotype = buildNet(784, 10)

        # for population_size in tqdm([10, 40, 100, 1000]):
        #     for mutation_power in [0.01, 0.03, 0.07, 0.1, 0.3, 0.7, 1.]:
        #         for mutation_rate in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]:
        #             for selection_proportion in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        #                 for batch_size in [10, 20, 40, 80, 100, 150, 200, 250]:
        #                     for initial_weight_range in [0.5, 1., 2., 4., 10., 20.]:

        # mutation_power = 0.03
        # mutation_rate = 0.04
        # selection_proportion = 0.4
        # initial_weight_range = 4
        # seed = 0
        file_identifier = "params_cosyne_%s_%s_%03d-%s" % (architecture, dataset_name, seed, str(params).replace("\t", "_"))
        print "Starting parameter sweep for cosyne %s" % file_identifier
        file_name = "%s.dat" % (file_identifier)
        f = open(file_name, 'w')
        f.write("seed\ttest_split\tvalidation_split")
        f.write("\tpopulation_size\tmutation_power\tmutation_rate\tselection_proportion\tbatch_size\tinitial_weight_range")
        f.write("\tevaluation_data\tevaluations\tfitness_type\tresult\n")

        # for population_size in tqdm([10, 40, 100, 400, 1000]):
        #     for batch_size in tqdm([10, 20, 40, 80, 100, 150, 200, 250]):
                # params = CosyneParams()
                # params.population_size = population_size
                # params.mutation_power = mutation_power
                # params.mutation_rate = mutation_rate
                # params.selection_proportion = selection_proportion
                # params.batch_size = batch_size
                # params.initial_weight_range = initial_weight_range


        # configure_for_training(params, max_evaluations, n_classes, eval_genotype, num_network_weights, seed, file_identifier)
        train_network = configure_for_training(params, max_evaluations, 10, eval_genotype, num_network_weights, seed, f)
        run_test_validate_splits(train_network, X_train, y_train, test_folds=3, validation_folds=5)

        f.close()



