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

def map_parts(genotype, layer1_size, layer2_size, last_index):
    current_index = last_index + layer1_size * layer2_size
    l_weights = genotype[last_index:current_index].reshape((layer1_size, layer2_size))
    last_index = current_index
    current_index += layer2_size
    b_weights = genotype[last_index:current_index]
    return l_weights, b_weights, current_index


def buildNet_perceptron(n_input, n_classes):
    y = tf.placeholder(tf.float64, [None, n_classes])
    x = tf.placeholder(tf.float64, [None, n_input])

    out = tf.placeholder(tf.float64, [n_input, n_classes])
    b_out = tf.placeholder(tf.float64, [n_classes])

    # Output layer with linear activation
    pred = tf.add(tf.matmul(x, out), b_out)

    # Define loss and accuracy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def runNetwork(sess, genotype, cur_data, cur_label):
        out_weights, b_out_weights, last_index = map_parts(genotype, n_input, n_classes, 0)
        assert(last_index == len(genotype))

        feed_dict = {x: cur_data, y: cur_label,
                    out: out_weights, b_out: b_out_weights}
        acc, c = sess.run([accuracy, cost], feed_dict=feed_dict)
        return acc, c

    return runNetwork


def buildNet_single_layer(n_hidden_1):
    def build_net(n_input, n_classes):
        y = tf.placeholder(tf.float64, [None, n_classes])
        x = tf.placeholder(tf.float64, [None, n_input])

        h1 = tf.placeholder(tf.float64, [n_input, n_hidden_1])
        out = tf.placeholder(tf.float64, [n_hidden_1, n_classes])
        b1 = tf.placeholder(tf.float64, [n_hidden_1])
        b_out = tf.placeholder(tf.float64, [n_classes])

        layer_1 = tf.add(tf.matmul(x, h1), b1)
        layer_1 = tf.nn.relu(layer_1)
        # Output layer with linear activation
        pred = tf.add(tf.matmul(layer_1, out), b_out)

        # Define loss and accuracy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        def runNetwork(sess, genotype, cur_data, cur_label):
            last_index = 0
            h1_weights, b1_weights, last_index = map_parts(genotype, n_input, n_hidden_1, last_index)
            out_weights, b_out_weights, last_index = map_parts(genotype, n_hidden_1, n_classes, last_index)
            assert(last_index == len(genotype))

            feed_dict = {x: cur_data, y: cur_label,
                        h1: h1_weights, out: out_weights,
                        b1: b1_weights, b_out: b_out_weights}
            acc, c = sess.run([accuracy, cost], feed_dict=feed_dict)
            return acc, c

        return runNetwork
    return build_net


def buildNet_three_layer(n_hidden_1, n_hidden_2, n_hidden_3):
    def build_net(n_input, n_classes):
        y = tf.placeholder(tf.float64, [None, n_classes])
        x = tf.placeholder(tf.float64, [None, n_input])

        h1 = tf.placeholder(tf.float64, [n_input, n_hidden_1])
        h2 = tf.placeholder(tf.float64, [n_hidden_1, n_hidden_2])
        h3 = tf.placeholder(tf.float64, [n_hidden_2, n_hidden_3])
        out = tf.placeholder(tf.float64, [n_hidden_3, n_classes])
        b1 = tf.placeholder(tf.float64, [n_hidden_1])
        b2 = tf.placeholder(tf.float64, [n_hidden_2])
        b3 = tf.placeholder(tf.float64, [n_hidden_3])
        b_out = tf.placeholder(tf.float64, [n_classes])

        layer_1 = tf.add(tf.matmul(x, h1), b1)
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, h2), b2)
        layer_2 = tf.nn.relu(layer_2)
        layer_3 = tf.add(tf.matmul(layer_2, h3), b3)
        layer_3 = tf.nn.relu(layer_3)
        # Output layer with linear activation
        pred = tf.add(tf.matmul(layer_3, out), b_out)

        # Define loss and accuracy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        def runNetwork(sess, genotype, cur_data, cur_label):
            # Layer 1
            last_index = 0
            h1_weights, b1_weights, last_index = map_parts(genotype, n_input, n_hidden_1, last_index)
            h2_weights, b2_weights, last_index = map_parts(genotype, n_hidden_1, n_hidden_2, last_index)
            h3_weights, b3_weights, last_index = map_parts(genotype, n_hidden_2, n_hidden_3, last_index)
            out_weights, b_out_weights, last_index = map_parts(genotype, n_hidden_3, n_classes, last_index)
            assert(last_index == len(genotype))

            feed_dict = {x: cur_data, y: cur_label,
                        h1: h1_weights, b1: b1_weights,
                        h2: h2_weights, b2: b2_weights,
                        h3: h3_weights, b3: b3_weights,
                        out: out_weights, b_out: b_out_weights}
            acc, c = sess.run([accuracy, cost], feed_dict=feed_dict)
            return acc, c

        return runNetwork
    return build_net


def buildNet_four_layer(n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4):
    def build_net(n_input, n_classes):
        y = tf.placeholder(tf.float64, [None, n_classes])
        x = tf.placeholder(tf.float64, [None, n_input])

        h1 = tf.placeholder(tf.float64, [n_input, n_hidden_1])
        h2 = tf.placeholder(tf.float64, [n_hidden_1, n_hidden_2])
        h3 = tf.placeholder(tf.float64, [n_hidden_2, n_hidden_3])
        h4 = tf.placeholder(tf.float64, [n_hidden_3, n_hidden_4])
        out = tf.placeholder(tf.float64, [n_hidden_4, n_classes])
        b1 = tf.placeholder(tf.float64, [n_hidden_1])
        b2 = tf.placeholder(tf.float64, [n_hidden_2])
        b3 = tf.placeholder(tf.float64, [n_hidden_3])
        b4 = tf.placeholder(tf.float64, [n_hidden_4])
        b_out = tf.placeholder(tf.float64, [n_classes])

        layer_1 = tf.add(tf.matmul(x, h1), b1)
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, h2), b2)
        layer_2 = tf.nn.relu(layer_2)
        layer_3 = tf.add(tf.matmul(layer_2, h3), b3)
        layer_3 = tf.nn.relu(layer_3)
        layer_4 = tf.add(tf.matmul(layer_3, h4), b4)
        layer_4 = tf.nn.relu(layer_4)
        # Output layer with linear activation
        pred = tf.add(tf.matmul(layer_4, out), b_out)

        # Define loss and accuracy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        def runNetwork(sess, genotype, cur_data, cur_label):
            # Layer 1
            last_index = 0
            h1_weights, b1_weights, last_index = map_parts(genotype, n_input, n_hidden_1, last_index)
            h2_weights, b2_weights, last_index = map_parts(genotype, n_hidden_1, n_hidden_2, last_index)
            h3_weights, b3_weights, last_index = map_parts(genotype, n_hidden_2, n_hidden_3, last_index)
            h4_weights, b4_weights, last_index = map_parts(genotype, n_hidden_3, n_hidden_4, last_index)
            out_weights, b_out_weights, last_index = map_parts(genotype, n_hidden_4, n_classes, last_index)
            assert(last_index == len(genotype))

            feed_dict = {x: cur_data, y: cur_label,
                        h1: h1_weights, b1: b1_weights,
                        h2: h2_weights, b2: b2_weights,
                        h3: h3_weights, b3: b3_weights,
                        h4: h4_weights, b4: b4_weights,
                        out: out_weights, b_out: b_out_weights}
            acc, c = sess.run([accuracy, cost], feed_dict=feed_dict)
            return acc, c

        return runNetwork
    return build_net


def createArchitecture(choice, dataset_name):
    if choice == "perceptron":
        if dataset_name == "mnist":
            return buildNet_perceptron, calc_num_weights([784, 10])
        else:
            return buildNet_perceptron, calc_num_weights([2, 2])
    elif choice == "small":
        if dataset_name == "mnist":
            return buildNet_single_layer(80), calc_num_weights([784, 80, 10])
        else:
            return buildNet_single_layer(5), calc_num_weights([2, 5, 2])
    elif choice == "big":
        if dataset_name == "mnist":
            return buildNet_single_layer(400), calc_num_weights([784, 400, 10])
        else:
            return buildNet_single_layer(10), calc_num_weights([2, 10, 2])
    elif choice == "deep":
        if dataset_name == "mnist":
            return buildNet_three_layer(40, 30, 20), calc_num_weights([784, 40, 30, 20, 10])
        else:
            return buildNet_four_layer(8, 8, 4, 4), calc_num_weights([2, 8, 8, 4, 4, 2])
    else:
        print "Bad luck no known architecture"