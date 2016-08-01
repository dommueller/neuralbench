from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer
from pybrain.structure import SoftmaxLayer

from sklearn.metrics import log_loss, accuracy_score

from dataset.create import createDataSet

import sys
import numpy as np

def buildNet_perceptron_mnist():
    return buildNetwork(784, 10, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)

def buildNet_small_layer_mnist():
    return buildNetwork(784, 80, 10, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)

def buildNet_big_layer_mnist():
    return buildNetwork(784, 400, 10, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)

def buildNet_deep_mnist():
    return buildNetwork(784, 40, 30, 20, 10, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=True)

def testNetwork(X, y, params, buildNet, file):
    nn = buildNet()
    nn._setParameters(np.array(params))

    results = []
    for example in X:
        result = nn.activate(example)
        results.append(result)
        nn.reset()

    predictions = [np.argmax(result) for result in results]

    acc =  accuracy_score(y, predictions)
    cost =  log_loss(y, results)

    file.write("%.2f\t%.2f\t%d\t%s\n" % (acc, cost, params.shape[0], "\t".join(("%.3f" % x for x in params))))

cases = [("perceptron", buildNet_perceptron_mnist), ("small", buildNet_small_layer_mnist),
                ("big", buildNet_big_layer_mnist), ("deep", buildNet_deep_mnist)]

if __name__=="__main__":
    case = cases[int(sys.argv[1])]
    seed = int(sys.argv[2])
    np.random.seed(seed)

    _, _, X_test, y_test = createDataSet("mnist")

    f = open("random.weights."+ case[0] + "." + str(seed), 'w')
    shape = case[1]().params.shape
    for _ in xrange(10000):
        params = np.random.uniform(-4, 4, shape)
        testNetwork(X_test, y_test, params, case[1], f)
    f.close()