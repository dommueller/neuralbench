from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer
from pybrain.structure import SoftmaxLayer
from pybrain.optimization import SNES, CMAES

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

def get_population_size(learned, cmaes):
    if cmaes:
        l = CMAES(lambda x: None, learned, verbose=False)
        return l.batchSize
    else:
        l = SNES(lambda x: None, learned, verbose=False)
        return l.batchSize

def configure_for_training(batch_size, max_evaluations, n_classes, buildNet, seed, f, cmaes):
    from sklearn.metrics import log_loss, accuracy_score

    def test_network(X, y, learned_params, evals, file_start, test_set):
        nn = buildNet(X.shape[1], n_classes)
        nn._setParameters(np.array(learned_params))

        results = []
        for example in X:
            result = nn.activate(example)
            results.append(result)
            nn.reset()

        predictions = [np.argmax(result) for result in results]

        acc =  accuracy_score(y, predictions)
        cost =  log_loss(y, results)

        f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, test_set, evals, "acc", acc))
        f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, test_set, evals, "cost", cost))


    def train_network(X_train, y_train, X_validate, y_validate, X_test, y_test, test_split=0, validate_split=0):
        file_start = "%d\t%d\t%d" % (seed, test_split, validate_split)

        n = buildNet(X_train.shape[1], n_classes)
        learned = n.params
        population_size = get_population_size(learned, cmaes)

        evaluations_per_generation = population_size * batch_size
        num_generations = max_evaluations/(evaluations_per_generation) + 1

        # Used to sample a batch with same class ratios
        from sklearn.cross_validation import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(y_train.reshape(-1), num_generations, train_size=batch_size, random_state=seed)
        train_indices = [batch_index for (batch_index, _) in sss]

        def objF(params):
            nn = buildNet(X_train.shape[1], n_classes)
            nn._setParameters(np.array(params))

            cur_data = X_train[train_indices[l.numLearningSteps]]
            cur_label = y_train[train_indices[l.numLearningSteps]]

            results = []
            for example, cor in zip(cur_data, cur_label):
                results.append(nn.activate(example))
                nn.reset()

            loss = log_loss(cur_label, results)
            return loss

        test_network(X_validate, y_validate, learned, 0, file_start, "val")
        test_network(X_test, y_test, learned, 0, file_start, "test")


        l = SNES(objF, learned, verbose=False)
        if cmaes:
            l = CMAES(objF, learned, verbose=False)
        l.minimize = True
        l.maxEvaluations = num_generations * population_size

        for generation in xrange(num_generations):
            result = l.learn(additionalLearningSteps=1)
            learned = result[0]

            train_evaluations = (generation + 1) * evaluations_per_generation

            test_network(X_train[train_indices[generation]], y_train[train_indices[generation]], learned, train_evaluations, file_start, "train")
            test_network(X_validate, y_validate, learned, train_evaluations, file_start, "val")
            test_network(X_test, y_test, learned, train_evaluations, file_start, "test")

            if generation % 100 == 0:
                f.flush()


    return train_network


def runExperiment(architecture, dataset, seed, max_evaluations, num_samples, cmaes=False):
    from neuralbench.classification.dataset.create import run_validate_splits
    np.random.seed(seed)
    file_name = "snes_%s_%s_%03d_e%010d_s%05d.dat" % (architecture, dataset["name"], seed, max_evaluations, num_samples)
    if cmaes:
       file_name = "cmaes_%s_%s_%03d_e%010d_s%05d.dat" % (architecture, dataset["name"], seed, max_evaluations, num_samples) 
    f = open(file_name, 'w')
    f.write("seed\ttest_split\tvalidation_split")
    f.write("\tevaluation_data\tevaluations\tfitness_type\tresult\n")

    buildNet = createArchitecture(architecture, dataset["name"])

    if dataset["name"] == "mnist":
        train_network = configure_for_training(num_samples, max_evaluations, 10, buildNet, seed, f, cmaes)
        run_validate_splits(train_network, dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"], folds=10, seed=seed)
    else:
        train_network = configure_for_training(num_samples, max_evaluations, 2, buildNet, seed, f, cmaes)
        run_validate_splits(train_network, dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"], folds=10, seed=seed)

    f.close()



