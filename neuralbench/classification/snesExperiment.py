from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer
from pybrain.structure import SoftmaxLayer
from pybrain.optimization import SNES

from sklearn.metrics import confusion_matrix
import numpy as np

def buildNet_perceptron(n_input, n_output):
    return buildNetwork(n_input, n_output, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)

def buildNet_small_layer(n_input, n_output):
    return buildNetwork(n_input, 5, n_output, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)

def buildNet_big_layer(n_input, n_output):
    return buildNetwork(n_input, 10, n_output, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)

def buildNet_deep(n_input, n_output):
    return buildNetwork(n_input, 8, 8, 4, 4, n_output, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)

def buildNet_small_layer_mnist(n_input, n_output):
    return buildNetwork(n_input, 80, n_output, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)

def buildNet_big_layer_mnist(n_input, n_output):
    return buildNetwork(n_input, 400, n_output, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)

def buildNet_deep_mnist(n_input, n_output):
    return buildNetwork(n_input, 40, 30, 20, n_output, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)

def createArchitecture(choice, dataset_name):
    if choice == "perceptron":
        return buildNet_perceptron
    elif choice == "small":
        if dataset_name == "mnist":
            return buildNet_small_layer_mnist
        else:
            return buildNet_small_layer
    elif choice == "big":
        if dataset_name == "mnist":
            return buildNet_big_layer_mnist
        else:
            return buildNet_big_layer
    elif choice == "deep":
        if dataset_name == "mnist":
            return buildNet_deep_mnist
        else:
            return buildNet_deep
    else:
        print "Bad luck no known architecture"

def testNetwork(data, n_classes, params, buildNet, evals, file, seed):
    nn = buildNet(data["X_test"].shape[1], n_classes)
    nn._setParameters(np.array(params))

    predictions = []

    for example in data["X_test"]:
        result = nn.activate(example)
        guess = np.argmax(result)
        predictions.append(guess)
        nn.reset()

    cm = confusion_matrix(data["y_test"], predictions)
    corr = 0
    for i in xrange(n_classes):
        for j in xrange(n_classes):
            if i == j:
                corr += cm[i, j]

    acc =  corr/float(len(predictions))
    # dataset, architecture, seed, evals, acc
    file.write("snes %d %d %f\n" % (seed, evals, acc))

def trainNetwork(data, n_classes, buildNet, file, seed, max_evaluations, num_samples):
    # The training functions uses the average of the cumulated reward and maximum height as fitness
    X_train = data["X_train"]
    y_train = data["y_train"]


    def objF(params):
        nn = buildNet(X_train.shape[1], n_classes)
        nn._setParameters(np.array(params))

        random_state = np.random.get_state()
        np.random.seed(l.numLearningSteps)
        sampled_data = np.random.choice(len(X_train), num_samples, replace=False)
        np.random.set_state(random_state)
        cur_data = X_train[sampled_data]
        cur_label = y_train[sampled_data]

        cum_correct = 0

        for example, cor in zip(cur_data, cur_label):
            result = nn.activate(example)
            loss_sum = 0
            for q, out in enumerate(result):
                if q != cor:
                    loss_sum += max(0, out - result[int(cor)] + 1)
            # guess = np.argmax(result)
            #if guess == cor:
                #cum_correct += 1
            cum_correct += loss_sum
            nn.reset()

        return cum_correct

    # Build net for initial random params
    n = buildNet(X_train.shape[1], n_classes)
    learned = n.params

    testNetwork(data, n_classes, learned, buildNet, 0, file, seed)

    l = SNES(objF, learned, verbose=False)
    # l.batchSize = batch_size
    batch_size = l.batchSize
    l.maxEvaluations = max_evaluations
    l.minimize = True

    for i in xrange((max_evaluations/batch_size)):
        result = l.learn(additionalLearningSteps=1)
        learned = result[0]

        testNetwork(data, n_classes, learned, buildNet, num_samples * (i + 1) * batch_size, file, seed)

    return learned

def runExperiment(architecture, dataset, seed, max_evaluations, num_samples):
    np.random.seed(seed)
    file_name = "snes_%s_%s_%03d_e%10d_s%05d.dat" % (architecture, dataset["name"], seed, max_evaluations, num_samples)
    f = open(file_name, 'w')
    buildNet = createArchitecture(architecture, dataset["name"])
    if dataset["name"] == "mnist":
        learned_params = trainNetwork(dataset, 10, buildNet, f, seed,
                max_evaluations=max_evaluations, num_samples=num_samples)
    else:
        learned_params = trainNetwork(dataset, 2, buildNet, f, seed,
                max_evaluations=max_evaluations, num_samples=num_samples)

    f.close()



