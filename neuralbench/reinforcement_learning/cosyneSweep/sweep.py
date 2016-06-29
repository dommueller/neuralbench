import numpy as np

import gym

from neuralbench.classification.cosyne.cosyne_params import CosyneParams
from neuralbench.reinforcement_learning.cosyneExperiment import net_configuration, build_simple_network, build_recurrent_network
from neuralbench.reinforcement_learning.cosyneExperiment import initialize_population, uniform_crossover, uniform_mutation
from neuralbench.reinforcement_learning.cosyneExperiment import permute_inplace, permute_inplace, create_new_generation

def train_network(env_name, step_limit, max_evaluations, build_network, params):
    env = gym.make(env_name)
    discrete_output = isinstance(env.action_space, gym.spaces.discrete.Discrete)

    def run_network(chromosome, current_seed=None, env=None):
        nn = build_network()
        nn._setParameters(np.array(chromosome)) 

        if env == None:
            env = gym.make(env_name)
        if current_seed != None:
            env.seed(current_seed)

        cum_reward = 0
        episode_count = 1

        for _ in xrange(episode_count):
            ob = env.reset()

            for _ in xrange(step_limit):
                result = nn.activate(np.reshape(ob, -1))
                
                if discrete_output:
                    action = np.argmax(result)
                else:
                    assert(len(result) % 2 == 0)
                    try:
                        action = np.array([np.random.normal(result[i], abs(result[i+1])) if result[i+1] != 0 else result[i] for i in xrange(0, len(result), 2)])
                    except:
                        print "Something went wrong with: ", result
                        action = np.random.uniform(0, 1)

                ob, reward, done, _ = env.step(action)
                cum_reward += reward
                if done:
                    break

            nn.reset()

        return cum_reward

    # Build initial population
    population = initialize_population(build_network, params)

    num_generations = max_evaluations/params.population_size + 1

    best = None
    for generation in xrange(num_generations):
        results = np.array([run_network(individium, current_seed=generation) for individium in population])
        sort_idx = np.argsort(results)[::-1]
        sorted_pop = population[sort_idx]
        sorted_results = results[sort_idx]
        best = sorted_pop[0]
        population = create_new_generation(sorted_pop, sorted_results, params)

    
    net = build_network()
    net._setParameters(best) 
    cum_reward = 0
    for run in xrange(100):
        ob = env.reset()

        for _ in xrange(step_limit):
            result = net.activate(np.reshape(ob, -1))

            if discrete_output:
                action = np.argmax(result)
            else:
                assert(len(result) % 2 == 0)
                action = np.array([np.random.normal(result[i], abs(result[i+1])) for i in xrange(0, len(result), 2)])

            ob, reward, done, _ = env.step(action)
            cum_reward += reward
            if done:
                break

        net.reset()

    return cum_reward

def main(job_id, input_params):
    import sys
    import logging
    gym.undo_logger_setup()
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    datasets = ["CartPole-v0", "Acrobot-v0", "MountainCar-v0", "Pendulum-v0"]
    params = CosyneParams()
    params.population_size = input_params["population_size"]
    params.mutation_power = input_params["mutation_power"]
    params.mutation_rate = input_params["mutation_rate"]
    params.selection_proportion = input_params["selection_proportion"]
    params.initial_weight_range = input_params["initial_weight_range"]

    max_evaluations = 10
    architecture = "simple"
    network_size = 20

    cum_reward = 0
    for env_name in datasets:
        step_limit = gym.envs.registry.spec(env_name).timestep_limit
        build_network = net_configuration(architecture, network_size, env_name)
        reward = train_network(env_name, step_limit, max_evaluations, build_network, params)
        if env_name == "Pendulum-v0":
            reward /= 10
        print reward, reward/100
        cum_reward += reward

    return cum_reward

if __name__ == '__main__':
    params = {"population_size": 2, "mutation_power": 0.000001, "mutation_rate": 0.000001, "selection_proportion": 1., "initial_weight_range": 0.5}
    rewards =  main(1, params)
    print rewards, rewards/400
