import pickle
import numpy as np
import gym

import MultiNEAT as NEAT


params = NEAT.Parameters()
params.PopulationSize = 300
# params.DynamicCompatibility = True
# params.WeightDiffCoeff = 4.0
# params.CompatTreshold = 2.0
# params.YoungAgeTreshold = 15
# params.SpeciesMaxStagnation = 15
# params.OldAgeTreshold = 35
# params.MinSpecies = 5
# params.MaxSpecies = 10
# params.RouletteWheelSelection = False
params.RecurrentProb = 0.4
# params.OverallMutationRate = 0.8

# params.MutateWeightsProb = 0.90

# params.WeightMutationMaxPower = 2.5
# params.WeightReplacementMaxPower = 5.0
# params.MutateWeightsSevereProb = 0.5
# params.WeightMutationRate = 0.25

# params.MaxWeight = 8

# params.MutateAddNeuronProb = 0.03
# params.MutateAddLinkProb = 0.05
# params.MutateRemLinkProb = 0.0

# params.MinActivationA  = 4.9
# params.MaxActivationA  = 4.9

# params.ActivationFunction_SignedSigmoid_Prob = 1.0
# # params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
# params.ActivationFunction_Tanh_Prob = 1.0
# # params.ActivationFunction_TanhCubic_Prob = 0.0
# params.ActivationFunction_SignedStep_Prob = 1.0
# # params.ActivationFunction_UnsignedStep_Prob = 0.0
# params.ActivationFunction_SignedGauss_Prob = 1.0
# # params.ActivationFunction_UnsignedGauss_Prob = 0.0
# params.ActivationFunction_Abs_Prob = 1.0
# params.ActivationFunction_SignedSine_Prob = 1.0
# # params.ActivationFunction_UnsignedSine_Prob = 0.0
# params.ActivationFunction_Linear_Prob = 1.0
# params.ActivationFunction_Relu_Prob = 1.0
# # params.ActivationFunction_Softplus_Prob = 0.0

params.ActivationFunction_SignedSigmoid_Prob = 1.0
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
params.ActivationFunction_Tanh_Prob = 1.0
params.ActivationFunction_TanhCubic_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 1.0
params.ActivationFunction_UnsignedStep_Prob = 0.0
params.ActivationFunction_SignedGauss_Prob = 1.0
params.ActivationFunction_UnsignedGauss_Prob = 0.0
params.ActivationFunction_Abs_Prob = 1.0
params.ActivationFunction_SignedSine_Prob = 1.0
params.ActivationFunction_UnsignedSine_Prob = 0.0
params.ActivationFunction_Linear_Prob = 1.0
params.ActivationFunction_Relu_Prob = 1.0
params.ActivationFunction_Softplus_Prob = 0.0


def train_network(env_name, seed, step_limit, max_evaluations, f):
    num_generations = max_evaluations/params.PopulationSize + 1
    env = gym.make(env_name)
    discrete_output = isinstance(env.action_space, gym.spaces.discrete.Discrete)

    def run_network(genome, env, episode_count=1):
        net = NEAT.NeuralNetwork()
        genome.BuildPhenotype(net)

        cum_rewards = 0

        episode_count = 1
        for _ in xrange(episode_count):
            ob = env.reset()
            net.Flush()

            for _ in xrange(step_limit):
                net.Input(np.reshape(ob, -1))
                for _ in xrange(1):
                    net.Activate()
                o = net.Output()
                if discrete_output:
                    action = np.argmax(o)
                else:
                    assert(len(o) % 2 == 0)
                    action = np.array([np.random.normal(o[i], abs(o[i+1])) for i in xrange(0, len(o), 2)])
                ob, reward, done, _ = env.step(action)
                cum_rewards += reward
                
                if done:
                    break

        return cum_rewards




    # TODO: handle different input and output types Box/Discrete...
    if discrete_output:
        g = NEAT.Genome(0, len(np.reshape(env.observation_space.sample(), -1)), 0, env.action_space.n, False, 
                        NEAT.ActivationFunction.LINEAR, NEAT.ActivationFunction.RELU, 0, params)
    else:
        g = NEAT.Genome(0, len(np.reshape(env.observation_space.sample(), -1)), 0, 2*len(np.reshape(env.action_space.sample(), -1)), False, 
                NEAT.ActivationFunction.LINEAR, NEAT.ActivationFunction.RELU, 0, params)
    pop = NEAT.Population(g, params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    current_best = None

    for generation in range(num_generations):
        for i_episode, genome in enumerate(NEAT.GetGenomeList(pop)):
            env = gym.make(env_name)
            env.seed(i_episode)
            reward = run_network(genome, env, episode_count=1)

            genome.SetFitness(reward)

        # print('Generation: {}, max fitness: {}'.format(generation,
        #                     max((x.GetFitness() for x in NEAT.GetGenomeList(pop)))))
        current_best = pickle.dumps(pop.GetBestGenome())
        pop.Epoch()

    best_genome = pickle.loads(current_best)
    env = gym.make(env_name)
    for run in xrange(100):
        reward = run_network(best_genome, env)
        f.write("%03d\t%d\t%d\t%.3f\n" % (seed, (num_generations * params.PopulationSize), run, reward))


def runExperiment(env_name, dataset, seed, step_limit, max_evaluations):
    np.random.seed(seed)
    file_name = "neat_neat_%s_%03d.dat" % (dataset, seed)
    f = open(file_name, 'w')
    f.write("seed\tevaluations\trun\tresult\n")

    # TODO run experiment
    train_network(env_name, seed, step_limit, max_evaluations, f)

    f.close()


