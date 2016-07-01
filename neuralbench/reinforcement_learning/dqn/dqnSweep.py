from dqnExperiment import DQN, DQN_continous, train, test
from dqn_params import DqnParams

import gym

def eval(input_params, network_size):
    import logging
    gym.undo_logger_setup()
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    datasets = ["CartPole-v0", "Acrobot-v0", "MountainCar-v0", "Pendulum-v0"]

    max_evaluations = 5000
    architecture = "simple"
    network_size = 20

    total_reward = 0
    for env_name in datasets:
        env = gym.make(env_name)
        if env_name == "Pendulum-v0":
            agent = DQN_continous(env, network_size, params, max_evaluations)
        else:
            agent = DQN(env, network_size, params, max_evaluations)

        # Train for max_evaluations episodes
        for train_i in xrange(max_evaluations):
            train(agent, env)

        cum_reward = 0
        # Test for 100 episodes
        for test_i in xrange(NUM_TEST_RUNS):
            cum_reward += test(agent, env)

        if env_name == "Pendulum-v0":
            cum_reward /= 10
        print cum_reward, cum_reward/100
        total_reward += cum_reward

    return -total_reward



if __name__ == '__main__':
    import sys
    import random
    from tqdm import *

    for network_size in tqdm([1, 5, 10, 40, 100, 300]):
        for seed in tqdm(xrange(100)):
            params = DqnParams()
            params.random_initialization(seed = seed)
            file_identifier = "params_dqn_simple_%d_%03d-%s" % (network_size, seed, str(params).replace("\t", "_"))
            # print "Starting parameter sweep for dqn %s" % file_identifier
            file_name = "%s.dat" % (file_identifier)
            f = open(file_name, 'w')
            f.write("result\tseed\tnetwork_size")
            f.write("\tgamma\tinitial_epsilon\tfinal_epsilon\treplay_size\tbatch_size\n")

            result = eval(params, network_size)
            f.write("%.3f\t%d\t%d\t%s\n" % (result, seed, network_size, str(params)))

            f.close()