import itertools
import tensorflow as tf
import numpy as np

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

def build_perceptron_network(net_architecture):
    assert(len(net_architecture) == 2)
    y = tf.placeholder(tf.float64, [None, net_architecture[-1]])
    x = tf.placeholder(tf.float64, [None, net_architecture[0]])


    # print "Creating w0 and b0, with a: %d and b: %d" % (net_architecture[0], net_architecture[1])
    w0 = tf.placeholder(tf.float64, [net_architecture[0], net_architecture[1]])
    b0 = tf.placeholder(tf.float64, [net_architecture[1]])

    # print "Creating pred with (x * w0 + b0)"
    pred = tf.add(tf.matmul(x, w0), b0)

    # Define loss and accuracy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def runNetwork(sess, genotype, cur_data, cur_label):
        feed_dict = {x: cur_data, y: cur_label}
        feed_dict[w0] = genotype[0][:-1]
        feed_dict[b0] = genotype[0][-1:].reshape(net_architecture[1])

        acc, c = sess.run([accuracy, cost], feed_dict=feed_dict)
        return acc, c

    genotype_shape = np.array([ np.arange(a*b + b).reshape((a+1,b)) for (a, b) in pairwise(net_architecture)])
    return runNetwork, genotype_shape


def build_multilayer_network(net_architecture):
    y = tf.placeholder(tf.float64, [None, net_architecture[-1]])
    x = tf.placeholder(tf.float64, [None, net_architecture[0]])

    layers = {}
    for i, (a, b) in enumerate(pairwise(net_architecture)):
        # print "Creating w%d and b%d, with a: %d and b: %d" % (i, i, a, b)
        layers["w%d" % i] = tf.placeholder(tf.float64, [a, b])
        layers["b%d" % i] = tf.placeholder(tf.float64, [b])

    # tanh((x * W_0) + b_0)
    # print "Creating l0 with tanh(x * w0 + b0)"
    layers["l0"] = tf.nn.tanh(tf.add(tf.matmul(x, layers["w0"]), layers["b0"]))

    hidden_layer = len(net_architecture) - 2

    for i in xrange(1, (hidden_layer)):
        # tanh((l_i-1 * W_i) + b_i)
        last_layer = i - 1
        # print "Creating l%d with tanh(l%d * w%d + b%d)" % (i, last_layer, i, i)
        layers["l%d" % i] = tf.nn.tanh(tf.add(tf.matmul(layers["l%d" % last_layer], layers["w%d" % i]), layers["b%d" % i]))

    # print "Creating pred with (l%d * w%d + b%d)" % (hidden_layer-1, hidden_layer, hidden_layer)
    pred = tf.add(tf.matmul(layers["l%d" % (hidden_layer-1)], layers["w%d" % hidden_layer]), layers["b%d" % hidden_layer])

    # Define loss and accuracy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def runNetwork(sess, genotype, cur_data, cur_label):
        feed_dict = {x: cur_data, y: cur_label}
        for i, n_nodes in enumerate(net_architecture[1:]):
            # print genotype[i].shape, net_architecture[i], n_nodes
            assert(genotype[i].shape[0] == net_architecture[i] + 1)
            assert(genotype[i].shape[1] == n_nodes)
            feed_dict[layers["w%d" % i]] = genotype[i][:-1]
            feed_dict[layers["b%d" % i]] = genotype[i][-1:].reshape(n_nodes)

        acc, c = sess.run([accuracy, cost], feed_dict=feed_dict)
        return acc, c

    genotype_shape = np.array([ np.arange(a*b + b, dtype=np.float64).reshape((a+1,b)) for (a, b) in pairwise(net_architecture)])
    return runNetwork, genotype_shape

def createArchitecture(choice, dataset_name):
    if choice == "perceptron":
        if dataset_name == "mnist":
            return build_perceptron_network([784, 10])
        else:
            return build_perceptron_network([2, 2])
    elif choice == "small":
        if dataset_name == "mnist":
            return build_multilayer_network([784, 80, 10])
        else:
            return build_multilayer_network([2, 5, 2])
    elif choice == "big":
        if dataset_name == "mnist":
            return build_multilayer_network([784, 400, 10])
        else:
            return build_multilayer_network([2, 10, 2])
    elif choice == "deep":
        if dataset_name == "mnist":
            return build_multilayer_network([784, 40, 30, 20, 10])
        else:
            return build_multilayer_network([2, 8, 8, 4, 4, 2])
    else:
        print "Bad luck no known architecture"


if __name__ == '__main__':
    arch = [4, 2, 2, 3]
    evalNet, weights = build_multilayer_network(arch)
    # for w in weights:
        # print w.shape

    with tf.Session() as sess:
        cur_data = np.arange(arch[0]*4).reshape((4,arch[0]))
        cur_label = np.arange(arch[-1]*4).reshape((4,arch[-1]))

        print evalNet(sess, weights, cur_data, cur_label)