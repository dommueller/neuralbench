import itertools
import tensorflow as tf

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

def calc_num_weights(network):
    weights = 0
    for a, b in pairwise(network):
        # Interlayer weights
        weights += a * b
    
    # Bias weights
    weights += sum(network[1:])
    return weights

def buildNet(network_layout):
    def netBuilder(x, genotype):
        # print "Genotype", len(genotype)
        assert(len(genotype) == calc_num_weights(network_layout))

        layers = {}
        last_index = 0

        for i, (a, b) in enumerate(pairwise(network_layout)):
            current_index = last_index + a * b
            weights = tf.Variable(genotype[last_index:current_index].reshape((a, b)))
            last_index = current_index
            current_index = last_index + b
            biases = tf.Variable(genotype[last_index:current_index])
            last_index = current_index
            if i == 0:
                layers["l%d" % i] = tf.nn.tanh(tf.add(tf.matmul(x, weights), biases))
            elif i == len(network_layout) - 2:
                layers["out"] = tf.add(tf.matmul(layers["l%d" % (i-1)], weights), biases)
            else:
                layers["l%d" % i] = tf.nn.tanh(tf.add(tf.matmul(layers["l%d" % (i-1)], weights), biases))

        return layers["out"]

    return netBuilder, calc_num_weights(network_layout)

def buildNet_perceptron(n_input, n_classes):
    def netBuilder(x, genotype):
        assert(len(genotype) == calc_num_weights([n_input, n_classes]))
        weights = tf.Variable(genotype[0:n_input * n_classes].reshape((n_input, n_classes)))
        biases = tf.Variable(genotype[n_input * n_classes:n_input * n_classes + n_classes])

        # Output layer with linear activation
        out_layer = tf.matmul(x, weights) + biases
        return out_layer

    return netBuilder, calc_num_weights([n_input, n_classes])

def createArchitecture(choice, dataset_name):
    if choice == "perceptron":
        if dataset_name == "mnist":
            return buildNet_perceptron(784, 10)
        else:
            return buildNet_perceptron(2, 2)
    elif choice == "small":
        if dataset_name == "mnist":
            return buildNet([784, 80, 10])
        else:
            return buildNet([2, 5, 2])
    elif choice == "big":
        if dataset_name == "mnist":
            return buildNet([784, 400, 10])
        else:
            return buildNet([2, 10, 2])
    elif choice == "deep":
        if dataset_name == "mnist":
            return buildNet([784, 40, 30, 20, 10])
        else:
            return buildNet([2, 8, 8, 4, 4, 2])
    else:
        print "Bad luck no known architecture"