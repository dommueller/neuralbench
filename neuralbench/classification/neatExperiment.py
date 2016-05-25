from sklearn.metrics import confusion_matrix
import numpy as np

import MultiNEAT as NEAT
import pickle

params = NEAT.Parameters()
# params.PopulationSize = 150
# params.DynamicCompatibility = True
# params.WeightDiffCoeff = 4.0
# params.CompatTreshold = 2.0
# params.YoungAgeTreshold = 15
# params.SpeciesMaxStagnation = 15
# params.OldAgeTreshold = 35
# params.MinSpecies = 5
# params.MaxSpecies = 10
# params.RouletteWheelSelection = False
# params.RecurrentProb = 0.4
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

params.ActivationFunction_SignedSigmoid_Prob = 1.0
params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
params.ActivationFunction_Tanh_Prob = 1.0
params.ActivationFunction_TanhCubic_Prob = 1.0
params.ActivationFunction_SignedStep_Prob = 1.0
params.ActivationFunction_UnsignedStep_Prob = 1.0
params.ActivationFunction_SignedGauss_Prob = 1.0
params.ActivationFunction_UnsignedGauss_Prob = 1.0
params.ActivationFunction_Abs_Prob = 1.0
params.ActivationFunction_SignedSine_Prob = 1.0
params.ActivationFunction_UnsignedSine_Prob = 1.0
params.ActivationFunction_Linear_Prob = 1.0
params.ActivationFunction_Relu_Prob = 1.0
params.ActivationFunction_Softplus_Prob = 1.0

def testNetwork(data, n_classes, genome, evals, file, seed):
    genome = pickle.loads(genome)
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    predictions = []

    for example in data["X_test"]:
        net.Flush()
        net.Input(example)
        for _ in range(3):
            net.Activate()

        result = softmax(net.Output())
        guess = np.argmax(result)
        predictions.append(guess)

    cm = confusion_matrix(data["y_test"], predictions)
    corr = 0
    for i in xrange(n_classes):
        for j in xrange(n_classes):
            if i == j:
                corr += cm[i, j]

    acc =  corr/float(len(predictions))
    # dataset, architecture, seed, evals, acc
    file.write("neat %d %d %f\n" % (seed, evals, acc))

def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

def trainNetwork(data, n_classes, file, seed, max_evaluations, num_samples, batch_size=150):
    params.PopulationSize = batch_size
    X_train = data["X_train"]
    y_train = data["y_train"]

    def objF(genome):
        net = NEAT.NeuralNetwork()
        genome.BuildPhenotype(net)

        random_state = np.random.get_state()
        np.random.seed(generation)
        sampled_data = np.random.choice(len(X_train), num_samples, replace=False)
        np.random.set_state(random_state)
        cur_data = X_train[sampled_data]
        cur_label = y_train[sampled_data]

        cum_correct = 0

        for example, cor in zip(cur_data, cur_label):
            net.Flush()
            net.Input(example)
            for _ in range(3):
                net.Activate()

            result = softmax(net.Output())
            loss_sum = 0
            for q, out in enumerate(result):
                if q != cor:
                    loss_sum += max(0, out - result[int(cor)] + 1)
            # Could also train with all results (e.g. l2 or logloss)
            #if guess == cor:
                #cum_correct += 1
            cum_correct += loss_sum

        # Return negative loss because fitness is expected
        return -cum_correct

    g = NEAT.Genome(0, X_train.shape[1], 0, n_classes, False, 
                    NEAT.ActivationFunction.LINEAR, NEAT.ActivationFunction.TANH, 0, params)
    pop = NEAT.Population(g, params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    current_best = None
    for generation in range((max_evaluations/batch_size)):
        for i_episode, genome in enumerate(NEAT.GetGenomeList(pop)):
            reward = objF(genome)

            genome.SetFitness(reward)

        current_best = pickle.dumps(pop.GetBestGenome())
        testNetwork(data, n_classes, current_best, num_samples * (generation + 1) * batch_size, file, seed)
        pop.Epoch()



def runExperiment(dataset, seed, max_evaluations, num_samples):
    np.random.seed(seed)
    file_name = "neat_neat_%s_%03d_e%06d_s%05d.dat" % (dataset["name"], seed, max_evaluations, num_samples)
    f = open(file_name, 'w')
    if dataset["name"] == "mnist":
        trainNetwork(dataset, 10, f, seed,
                max_evaluations=max_evaluations, num_samples=num_samples, batch_size=150)
    else:
        trainNetwork(dataset, 2, f, seed,
                max_evaluations=max_evaluations, num_samples=num_samples, batch_size=150)

    f.close()

