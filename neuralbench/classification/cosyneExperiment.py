import numpy as np
import tensorflow as tf
import cosyneNetworks

# Taken from 
# https://github.com/tensorflow/tensorflow/blob/1d76583411038767f673a0c96174c80eaf9ff42f/tensorflow/g3doc/tutorials/mnist/input_data.py
def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
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

def normal_mutation(parent, mutation_probability):
    mutation_indices = np.random.uniform(0, 1, parent.shape) >= mutation_probability
    mutations = np.random.normal(0, 0.1, parent.shape)
    mutations[mutation_indices] = 0
    return parent + mutations

# Shuffles weights in one subpoplation / shuffles columns
def permute_inplace(population):
    transposed = population.T
    map(np.random.shuffle, transposed)

def createNewPop(old_pop, results):
    top_idx = int(len(old_pop) * 0.4)
    elite = old_pop[0:top_idx]
    offspring = []
    needed_offsprings = len(old_pop) - top_idx
    for _ in xrange(needed_offsprings/2 + 1):
        # parents = np.random.choice(len(elite), 2, replace=False, p=softmax(-np.array(results[0:top_idx])))
        # Substract max, so that lowest will have highest prob
        new_results = np.array(results[0:top_idx]) - (results[-1] + 1)
        probs = new_results / sum(new_results)
        parents = np.random.choice(len(elite), 2, replace=False, p=probs)
        child1, child2 = uniform_crossover(elite[parents[0]], elite[parents[1]])
        child1 = normal_mutation(child1, 0.03)
        child2 = normal_mutation(child2, 0.03)
        offspring.append(child1)
        offspring.append(child2)

    permute_inplace(elite)
    new_pop = old_pop[:]
    new_pop[0:top_idx] = [normal_mutation(el, 0.1) for el in elite]
    new_pop[top_idx:] = offspring[0:needed_offsprings]

    assert(len(old_pop) == len(new_pop))

    return new_pop


def eval_genotype(buildNet, genotype, cur_data, cur_label):
    "Takes a genotype, builds the genotype and evaluates it on the given data"

    y = tf.placeholder(tf.float64, [None, cur_label.shape[1]])
    x = tf.placeholder(tf.float64, [None, cur_data.shape[1]])

    pred = buildNet(x, genotype)

    # Define loss and accuracy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Initializing the variables
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        c = sess.run(cost, feed_dict={x: cur_data, y: cur_label})
        acc = accuracy.eval({x: cur_data, y: cur_label})
        return acc, c

def trainNetwork(data, n_classes, buildNet, num_network_weights, file, seed, max_evaluations, num_samples):
    X_train = data["X_train"]
    y_train = dense_to_one_hot(data["y_train"], n_classes)
    X_test = data["X_test"]
    y_test = dense_to_one_hot(data["y_test"], n_classes)

    batch_size = 40
    pop = initialize_population(num_network_weights, batch_size, 1)

    for i in xrange((max_evaluations/batch_size)):
    # for i in xrange(10):
        sampled_data = np.random.choice(len(X_train), num_samples, replace=False)
        cur_data = X_train[sampled_data]
        cur_label = y_train[sampled_data]

        results = np.array([cost for _, cost in [eval_genotype(buildNet, pheno, cur_data, cur_label) for pheno in pop]])
        # print "Results", results
        # print "Min", np.argmin(results)
        acc, _ = eval_genotype(buildNet, pop[np.argmin(results)], X_test, y_test)
        evals = (i + 1) * num_samples * batch_size
        file.write("cosyne %d %d %f\n" % (seed, evals, acc))
        # Sort population by cost
        sort_idx = np.argsort(results)
        sorted_pop = pop[sort_idx]
        sorted_results = results[sort_idx]

        # print "Population"
        # print sorted_pop
        pop = createNewPop(sorted_pop, sorted_results)
        # print pop

        # create new 


def runExperiment(architecture, dataset, seed, max_evaluations, num_samples):
    np.random.seed(seed)
    file_name = "cosyne_%s_%s_%03d_e%010d_s%05d.dat" % (architecture, dataset["name"], seed, max_evaluations, num_samples)
    f = open(file_name, 'w')
    buildNet, num_network_weights = cosyneNetworks.createArchitecture(architecture, dataset["name"])
    if dataset["name"] == "mnist":
        trainNetwork(dataset, 10, buildNet, num_network_weights, f, seed,
                max_evaluations=max_evaluations, num_samples=num_samples)
    else:
        trainNetwork(dataset, 2, buildNet, num_network_weights, f, seed,
                max_evaluations=max_evaluations, num_samples=num_samples)

    f.close()