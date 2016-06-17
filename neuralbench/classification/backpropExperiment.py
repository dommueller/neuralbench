import numpy as np
import tensorflow as tf

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


def buildNet_deep(x, n_input, n_classes):

    n_hidden_1 = 8
    n_hidden_2 = 8
    n_hidden_3 = 4
    n_hidden_4 = 4

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.tanh(layer_3)
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.tanh(layer_4)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

def buildNet_big(x, n_input, n_classes):

    n_hidden_1 = 10

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

def buildNet_small(x, n_input, n_classes):

    n_hidden_1 = 5

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

def buildNet_deep_mnist(x, n_input, n_classes):

    n_hidden_1 = 40
    n_hidden_2 = 30
    n_hidden_3 = 20

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.tanh(layer_3)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

def buildNet_big_mnist(x, n_input, n_classes):

    n_hidden_1 = 400

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

def buildNet_small_mnist(x, n_input, n_classes):

    n_hidden_1 = 80

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

def buildNet_perceptron(x, n_input, n_classes):

    weights = {
        'out': tf.Variable(tf.random_normal([n_input, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Output layer with linear activation
    out_layer = tf.matmul(x, weights['out']) + biases['out']
    return out_layer

def createArchitecture(choice, dataset_name):
    if choice == "perceptron":
        return buildNet_perceptron
    elif choice == "small":
        if dataset_name == "mnist":
            return buildNet_small_mnist
        else:
            return buildNet_small
    elif choice == "big":
        if dataset_name == "mnist":
            return buildNet_big_mnist
        else:
            return buildNet_big
    elif choice == "deep":
        if dataset_name == "mnist":
            return buildNet_deep_mnist
        else:
            return buildNet_deep
    else:
        print "Bad luck no known architecture"

def testNetwork(pred, x, y, X_test, y_test, n_classes, evals, file, seed):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    acc = accuracy.eval({x: X_test, y: y_test})
    file.write("backprop %d %d %f\n" % (seed, evals, acc))


def trainNetwork(data, n_classes, buildNet, file, seed, max_evaluations, num_samples):
    learning_rate = 0.01 if data["name"] == "mmnist" else 0.03


    X_train = data["X_train"]
    y_train = dense_to_one_hot(data["y_train"], n_classes)
    X_test = data["X_test"]
    y_test = dense_to_one_hot(data["y_test"], n_classes)

    y = tf.placeholder("float", [None, n_classes])
    x = tf.placeholder("float", [None, X_train.shape[1]])

    pred = buildNet(x, X_train.shape[1], n_classes)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        testNetwork(pred, x, y, X_test, y_test, n_classes, 0 , file, seed)

        # Training cycle
        epochs = num_samples * max_evaluations / data["X_train"].shape[0]
        batches = data["X_train"].shape[0] / num_samples

        for epoch in xrange(epochs):
            # Shuffle data, so each epoch different samples are in one batch
            sampled_data = np.random.choice(len(X_train), len(X_train), replace=False)
            shuffled_data = X_train[sampled_data]
            shuffled_label = y_train[sampled_data]

            for i in xrange(batches):
                index = i * num_samples
                next_index = (i + 1) * num_samples
                cur_data = shuffled_data[index:next_index]
                cur_label = shuffled_label[index:next_index]

                _, c = sess.run([optimizer, cost], feed_dict={x: cur_data, y: cur_label})

            testNetwork(pred, x, y, X_test, y_test, n_classes, (epoch + 1) * batches * num_samples, file, seed)
            file.flush()

def configure_for_training(batch_size, max_evaluations, n_classes, buildNet, seed, f):

    def train_network(X_train, y_train, X_validate, y_validate, X_test, y_test, test_split=0, validate_split=0):
        learning_rate = 0.01
        file_start = "%d\t%d\t%d" % (seed, test_split, validate_split)

        assert(n_classes == len(np.unique(y_train)))
        y_train_one_hot = dense_to_one_hot(y_train, n_classes)
        assert(n_classes == len(np.unique(y_validate)))
        y_validate = dense_to_one_hot(y_validate, n_classes)
        assert(n_classes == len(np.unique(y_test)))
        y_test = dense_to_one_hot(y_test, n_classes)

        y = tf.placeholder("float", [None, n_classes])
        x = tf.placeholder("float", [None, X_train.shape[1]])

        pred = buildNet(x, X_train.shape[1], n_classes)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Initializing the variables
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            acc_train, c_train = sess.run([accuracy, cost], feed_dict={x: X_train, y: y_train_one_hot})
            acc_val, c_val = sess.run([accuracy, cost], feed_dict={x: X_validate, y: y_validate})
            acc_test, c_test = sess.run([accuracy, cost], feed_dict={x: X_test, y: y_test})

            f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "train", 0, "acc", acc_train))
            f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "val", 0, "acc", acc_val))
            f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "test", 0, "acc", acc_test))
            f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "train", 0, "cost", c_train))
            f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "val", 0, "cost", c_val))
            f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "test", 0, "cost", c_test))

            num_generations = max_evaluations/(batch_size) + 1

            # Used to sample a batch with same class ratios
            from sklearn.cross_validation import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(y_train.reshape(-1), num_generations, train_size=batch_size, random_state=seed)
            
            for generation, (batch_index, _) in enumerate(sss):
                X_current = X_train[batch_index]
                y_current = y_train_one_hot[batch_index]

                _, acc_train, c_train = sess.run([optimizer, accuracy, cost], feed_dict={x: X_current, y: y_current})
                acc_val, c_val = sess.run([accuracy, cost], feed_dict={x: X_validate, y: y_validate})
                acc_test, c_test = sess.run([accuracy, cost], feed_dict={x: X_test, y: y_test})
                
                train_evaluations = (generation + 1) * batch_size

                f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "train", train_evaluations, "acc", acc_train))
                f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "val", train_evaluations, "acc", acc_val))
                f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "test", train_evaluations, "acc", acc_test))
                f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "train", train_evaluations, "cost", c_train))
                f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "val", train_evaluations, "cost", c_val))
                f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "test", train_evaluations, "cost", c_test))
                if generation % 100 == 0:
                    f.flush()

    return train_network


def runExperiment(architecture, dataset, seed, max_evaluations, num_samples):
    from neuralbench.classification.dataset.create import run_validate_splits
    np.random.seed(seed)
    file_name = "backprop_%s_%s_%03d_e%010d_s%05d.dat" % (architecture, dataset["name"], seed, max_evaluations, num_samples)
    f = open(file_name, 'w')
    f.write("seed\ttest_split\tvalidation_split")
    f.write("\tevaluation_data\tevaluations\tfitness_type\tresult\n")

    subs = createArchitecture(architecture, dataset["name"])
    if dataset["name"] == "mnist":
        train_network = configure_for_training(num_samples, max_evaluations, 10, subs, seed, f)
        run_validate_splits(train_network, dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"], folds=10, seed=seed)
    else:
        train_network = configure_for_training(num_samples, max_evaluations, 2, subs, seed, f)
        run_validate_splits(train_network, dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"], folds=10, seed=seed)


    f.close()