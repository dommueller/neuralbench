#!/usr/bin/python

import argparse
from dataset.create import createDataSet

if __name__ == '__main__':
    algorithms = ["neat", "neattanh", "hyperneat", "hyperneattanh", "snes", "cmaes", "backprop", "cosyne", "leea"]
    datasets = ["spiral", "xor", "circle", "gaussian", "mnist"]
    architectures = ["perceptron", "small", "big", "deep"]

    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", help="the algorithm that should be used", choices=algorithms)
    parser.add_argument("dataset", help="the dataset that should be used", choices=datasets)
    parser.add_argument("seed", help="the seed that should be used", type=int)
    parser.add_argument("-a", "--architecture", help="the architecture that should be used if not neat", choices=architectures)
    parser.add_argument("-e", "--evaluations", help="max number of evaluations", type=int, default=10000)
    parser.add_argument("-s", "--samples", help="number of samples per evaluation", type=int, default=50)
    args = parser.parse_args()
    if args.architecture:
        print "Training on %s using %s and %s, the seed is %d (Using %d samples for %d evaluations)" % (args.dataset, args.algorithm, args.architecture, args.seed, args.samples, args.evaluations)
    else:
        print "Training on %s using %s and seed is %d (Using %d samples for %d evaluations)" % (args.dataset, args.algorithm, args.seed, args.samples, args.evaluations)

    X_train, y_train, X_test, y_test = createDataSet(args.dataset, args.seed)
    data = {"X_train": X_train, "y_train": y_train, "X_test": X_test , "y_test": y_test, "name": args.dataset}

    if args.algorithm == "neat":
        import neatExperiment
        neatExperiment.runExperiment(data, args.seed, args.evaluations, args.samples)
    elif args.algorithm == "neattanh":
        import neatExperiment
        neatExperiment.runExperiment(data, args.seed, args.evaluations, args.samples, tanh=True)
    elif args.algorithm == "hyperneat":
        import hyperNeatExperiment
        hyperNeatExperiment.runExperiment(args.architecture, data, args.seed, args.evaluations, args.samples)
    elif args.algorithm == "hyperneattanh":
        import hyperNeatExperiment
        hyperNeatExperiment.runExperiment(args.architecture, data, args.seed, args.evaluations, args.samples, tanh=True)
    elif args.algorithm == "snes":
        import snesExperiment
        snesExperiment.runExperiment(args.architecture, data, args.seed, args.evaluations, args.samples)
    elif args.algorithm == "cmaes":
        import snesExperiment
        snesExperiment.runExperiment(args.architecture, data, args.seed, args.evaluations, args.samples, cmaes=True)
    elif args.algorithm == "backprop":
        import backpropExperiment
        backpropExperiment.runExperiment(args.architecture, data, args.seed, args.evaluations, args.samples)
    elif args.algorithm == "cosyne":
        from neuralbench.classification.cosyne.cosyneExperiment import runExperiment
        runExperiment(args.architecture, data, args.seed, args.evaluations, args.samples)
    elif args.algorithm == "leea":
        from neuralbench.classification.leea.leeaExperiment import runExperiment
        runExperiment(args.architecture, data, args.seed, args.evaluations, args.samples)
    else:
        print "Algorithm not found"


