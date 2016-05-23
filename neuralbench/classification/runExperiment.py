#!/usr/bin/python

import tsPlaygroundDatasets
from sklearn.cross_validation import train_test_split
import argparse
import numpy as np

def createDataSet(choice, seed):
    if choice == "spiral":
        data = tsPlaygroundDatasets.spiralData(500, 0.25, seed)
        train, test = train_test_split(data, test_size = 0.5, random_state=seed)
        X_train = train.as_matrix(['x', 'y'])
        y_train = train.as_matrix(['label'])
        X_test = test.as_matrix(['x', 'y'])
        y_test = test.as_matrix(['label'])
        return (X_train, y_train, X_test, y_test)
    elif choice == "xor":
        data = tsPlaygroundDatasets.xorData(500, 0.0, seed)
        train, test = train_test_split(data, test_size = 0.5, random_state=seed)
        X_train = train.as_matrix(['x', 'y'])
        y_train = train.as_matrix(['label'])
        X_test = test.as_matrix(['x', 'y'])
        y_test = test.as_matrix(['label'])
        return (X_train, y_train, X_test, y_test)
    elif choice == "circle":
        data = tsPlaygroundDatasets.circleData(500, 0.0, seed)
        train, test = train_test_split(data, test_size = 0.5, random_state=seed)
        X_train = train.as_matrix(['x', 'y'])
        y_train = train.as_matrix(['label'])
        X_test = test.as_matrix(['x', 'y'])
        y_test = test.as_matrix(['label'])
        return (X_train, y_train, X_test, y_test)
    elif choice == "gaussian":
        data = tsPlaygroundDatasets.circleData(500, 0.0, seed)
        train, test = train_test_split(data, test_size = 0.5, random_state=seed)
        X_train = train.as_matrix(['x', 'y'])
        y_train = train.as_matrix(['label'])
        X_test = test.as_matrix(['x', 'y'])
        y_test = test.as_matrix(['label'])
        return (X_train, y_train, X_test, y_test)
    elif choice == "mnist":
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
        # rescale the data, use the traditional train/test split
        X, y = mnist.data / 255., mnist.target
        test_size = 60000
        X_train, X_test = X[:test_size], X[60000:]
        y_train, y_test = y[:test_size], y[60000:]
        return (X_train, y_train, X_test, y_test)
    else:
        print "Bad luck no known dataset"


if __name__ == '__main__':
    algorithms = ["neat", "snes", "backprop", "cosyne"]
    datasets = ["spiral", "xor", "circle", "gaussian", "mnist"]
    architectures = ["perceptron", "small", "big", "deep"]

    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", help="the algorithm that should be used", choices=algorithms)
    parser.add_argument("dataset", help="the dataset that should be used", choices=datasets)
    parser.add_argument("seed", help="the seed that should be used", type=int)
    parser.add_argument("-a", "--architecture", help="the architecture that should be used if not neat", choices=architectures)
    args = parser.parse_args()
    if args.architecture:
        print "Training on %s using %s and %s, the seed is %d" % (args.dataset, args.algorithm, args.architecture, args.seed)
    else:
        print "Training on %s using %s and seed is %d" % (args.dataset, args.algorithm, args.seed)

    X_train, y_train, X_test, y_test = createDataSet(args.dataset, args.seed)
    data = {"X_train": X_train, "y_train": y_train, "X_test": X_test , "y_test": y_test, "name": args.dataset}

    if args.algorithm == "neat":
        # TODO
        pass
    elif args.algorithm == "snes":
        import snesExperiment
        snesExperiment.runExperiment(args.architecture, data, args.seed)
    elif args.algorithm == "backprop":
        # TODO
        pass
    elif args.algorithm == "cosyne":
        # Not implemented yet
        pass
    else:
        print "Bad luck something went wrong"


