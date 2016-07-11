#!/usr/bin/python

import logging
import gym
import argparse

def createEnvironment(environment_name):
    if environment_name == "cartpole":
        return "CartPole-v0"
    elif environment_name == "acrobot":
        return "Acrobot-v0"
    elif environment_name == "mountaincar":
        return "MountainCar-v0"
    elif environment_name == "pendulum":
        return "Pendulum-v0"
    elif environment_name == "nchain":
        pass
        # return "NChain-v0"
    elif environment_name == "blackjack":
        pass
        # return "Blackjack-v0"
    elif environment_name == "go9":
        return "Go9x9-v0"
    elif environment_name == "go19":
        return "Go19x19-v0"


if __name__ == '__main__':
    gym.undo_logger_setup()
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)


    algorithms = ["neat", "hyperneat", "snes", "dqn", "cosyne", "leea"]
    datasets = ["cartpole", "acrobot", "mountaincar", "pendulum", "nchain", "blackjack", "go9", "go19"]
    architectures = ["simple", "recurrent"]

    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", help="the algorithm that should be used", choices=algorithms)
    parser.add_argument("dataset", help="the dataset that should be used", choices=datasets)
    parser.add_argument("seed", help="the seed that should be used", type=int)
    parser.add_argument("-a", "--architecture", help="the architecture that should be used if not neat", choices=architectures)
    parser.add_argument("-s", "--network_size", help="number of hidden units", type=int, default=40)
    parser.add_argument("-e", "--evaluations", help="max number of evaluations", type=int, default=10000)
    args = parser.parse_args()
    if args.architecture:
        print "Training on %s using %s and %s, the seed is %d (max evaluations: %d)" % (args.dataset, args.algorithm, args.architecture, args.seed, args.evaluations)
    else:
        print "Training on %s using %s and seed is %d (max evaluations: %d)" % (args.dataset, args.algorithm, args.seed, args.evaluations)

    env_name = createEnvironment(args.dataset)

    if args.algorithm == "neat":
        import neatExperiment
        neatExperiment.runExperiment(env_name, args.dataset, args.seed, args.evaluations, 20)
    elif args.algorithm == "hyperneat":
        import hyperNeatExperiment
        pass
    elif args.algorithm == "snes":
        import snesExperiment
        snesExperiment.runExperiment(env_name, args.dataset, args.architecture, args.network_size, args.seed, args.evaluations, 20)
    elif args.algorithm == "dqn":
        import dqn.dqnExperiment
        dqn.dqnExperiment.runExperiment(env_name, args.dataset, args.architecture, args.network_size, args.seed, args.evaluations, 20)
    elif args.algorithm == "cosyne":
        import cosyneExperiment
        cosyneExperiment.runExperiment(env_name, args.dataset, args.architecture, args.network_size, args.seed, args.evaluations, 20)
    elif args.algorithm == "leea":
        import leeaExperiment
        leeaExperiment.runExperiment(env_name, args.dataset, args.architecture, args.network_size, args.seed, args.evaluations, 20)
    else:
        print "Algorithm not found"


