import numpy as np

import gym

from neuralbench.classification.leea.leea_params import LeeaParams
from neuralbench.reinforcement_learning.leeaExperiment import net_configuration, build_simple_network, build_recurrent_network
from neuralbench.reinforcement_learning.leeaExperiment import Gene, initialize_population, sort_population, create_new_population

def train_network(env_name, step_limit, max_evaluations, build_network, params):
    env = gym.make(env_name)
    discrete_output = isinstance(env.action_space, gym.spaces.discrete.Discrete)

    def run_network(chromosome, env=None):
        nn = build_network()
        nn._setParameters(np.array(chromosome)) 

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
    population = initialize_population(build_network, run_network, params)

    num_generations = max_evaluations/params.population_size + 1
    elite_size = int(params.population_size * params.selection_proportion)
    elite_size = 2 if elite_size < 2 else elite_size

    best = None
    for generation in xrange(num_generations):
        for individual in population:
            env = gym.make(env_name)
            env.seed(generation)
            individual.calc_fitness(env, params)

        population = sort_population(population)
        best = population[0]
        population = create_new_population(params, population[:elite_size])
    

    env = gym.make(env_name)
    cum_reward = 0
    for run in xrange(100):
        cum_reward += best.evaluate(env)

    return cum_reward

def main(job_id, input_params):
    import sys
    import logging
    gym.undo_logger_setup()
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    datasets = ["CartPole-v0", "Acrobot-v0", "MountainCar-v0", "Pendulum-v0"]
    params = LeeaParams()
    params.parent_fitness_decay = input_params["parent_fitness_decay"][0]
    params.mutation_power_decay = 0.99
    params.sexual_reproduction_proportion = 0.5
    params.population_size = input_params["population_size"][0]
    params.starting_mutation_power = input_params["starting_mutation_power"][0]
    params.mutation_power = params.starting_mutation_power
    params.mutation_rate = input_params["mutation_rate"][0]
    params.selection_proportion = input_params["selection_proportion"][0]

    max_evaluations = 5000
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

    return -cum_reward

if __name__ == '__main__':
    params = {"initial_weight_range": 0.500000, "parent_fitness_decay": 0.010000, "population_size": 5, "starting_mutation_power": 0.000001, "mutation_rate": 0.000001, "selection_proportion": 0.010000}

    rewards =  main(1, params)
    print rewards, rewards/400