from sklearn.metrics import confusion_matrix
import numpy as np

import MultiNEAT as NEAT
import pickle


def buildNet_perceptron():
    substrate = NEAT.Substrate([(-1, -1), (-1, 1)],
                           [],
                           [(1, 0), (1, 1)])
    substrate.m_allow_input_output_links =True;
    substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.TANH;
    substrate.m_output_nodes_activation = NEAT.ActivationFunction.LINEAR;
    return substrate

def buildNet_small_layer():
    substrate = NEAT.Substrate([(-1, -1), (-1, 1)],
                           [(0, i - (5 - 1) / 2) for i in xrange(5)],
                           [(1, 0), (1, 1)])
    substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.TANH;
    substrate.m_output_nodes_activation = NEAT.ActivationFunction.LINEAR;
    return substrate

def buildNet_big_layer():
    substrate = NEAT.Substrate([(-1, -1), (-1, 1)],
                           [(0, i - (10 - 1) / 2) for i in xrange(10)],
                           [(1, 0), (1, 1)])
    substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.TANH;
    substrate.m_output_nodes_activation = NEAT.ActivationFunction.LINEAR;
    return substrate

def buildNet_perceptron_mnist():
    substrate = NEAT.Substrate([(-1, x, y) for x in xrange(28) for y in xrange(28)],
                           [],
                           [(1, 0, c) for x in xrange(10)])
    substrate.m_allow_input_output_links =True;
    substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.TANH;
    substrate.m_output_nodes_activation = NEAT.ActivationFunction.LINEAR;
    return substrate

def buildNet_small_layer_mnist():
    substrate = NEAT.Substrate([(-1, x, y) for x in xrange(28) for y in xrange(28)],
                           [(0, x, y) for x in xrange(8) for y in xrange(10)],
                           [(1, 0, c) for x in xrange(10)])
    substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.TANH;
    substrate.m_output_nodes_activation = NEAT.ActivationFunction.LINEAR;
    return substrate

def buildNet_big_layer_mnist():
    substrate = NEAT.Substrate([(-1, x, y) for x in xrange(28) for y in xrange(28)],
                           [(0, x, y) for x in xrange(20) for y in xrange(20)],
                           [(1, 0, c) for x in xrange(10)])
    substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.TANH;
    substrate.m_output_nodes_activation = NEAT.ActivationFunction.LINEAR;
    return substrate
    
def createArchitecture(choice, dataset_name):
    if choice == "perceptron":
        if dataset_name == "mnist":
            return buildNet_perceptron_mnist()
        else:
            return buildNet_perceptron()
    elif choice == "small":
        if dataset_name == "mnist":
            return buildNet_small_layer_mnist()
        else:
            return buildNet_small_layer()
    elif choice == "big":
        if dataset_name == "mnist":
            return buildNet_big_layer_mnist()
        else:
            return buildNet_big_layer()
    elif choice == "deep":
        raise NotImplementedError('Deep architectures are not implemented in HyperNEAT (multineat).')
    else:
        print "Bad luck no known architecture"

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

# params.ActivationFunction_SignedSigmoid_Prob = 1.0
# params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
# params.ActivationFunction_Tanh_Prob = 1.0
# params.ActivationFunction_TanhCubic_Prob = 1.0
# params.ActivationFunction_SignedStep_Prob = 1.0
# params.ActivationFunction_UnsignedStep_Prob = 1.0
# params.ActivationFunction_SignedGauss_Prob = 1.0
# params.ActivationFunction_UnsignedGauss_Prob = 1.0
# params.ActivationFunction_Abs_Prob = 1.0
# params.ActivationFunction_SignedSine_Prob = 1.0
# params.ActivationFunction_UnsignedSine_Prob = 1.0
# params.ActivationFunction_Linear_Prob = 1.0
# params.ActivationFunction_Relu_Prob = 1.0
# params.ActivationFunction_Softplus_Prob = 1.0

def set_activation_functions(tanh):
    if tanh:
        params.ActivationFunction_SignedSigmoid_Prob = 0.0
        params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
        params.ActivationFunction_Tanh_Prob = 1.0
        params.ActivationFunction_TanhCubic_Prob = 0.0
        params.ActivationFunction_SignedStep_Prob = 0.0
        params.ActivationFunction_UnsignedStep_Prob = 0.0
        params.ActivationFunction_SignedGauss_Prob = 0.0
        params.ActivationFunction_UnsignedGauss_Prob = 0.0
        params.ActivationFunction_Abs_Prob = 0.0
        params.ActivationFunction_SignedSine_Prob = 0.0
        params.ActivationFunction_UnsignedSine_Prob = 0.0
        params.ActivationFunction_Linear_Prob = 0.0
        params.ActivationFunction_Relu_Prob = 0.0
        params.ActivationFunction_Softplus_Prob = 0.0
    else:
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

def testNetwork(data, n_classes, subs, genome, evals, file, seed):
    genome = pickle.loads(genome)
    net = NEAT.NeuralNetwork()
    genome.BuildHyperNEATPhenotype(net, subs)

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

def trainNetwork(data, n_classes, subs, file, seed, max_evaluations, num_samples, batch_size=150):
    params.PopulationSize = batch_size
    X_train = data["X_train"]
    y_train = data["y_train"]

    def objF(genome):
        net = NEAT.NeuralNetwork()
        genome.BuildHyperNEATPhenotype(net, subs)

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
            # guess = np.argmax(result)
            # Could also train with all results (e.g. l2 or logloss)
            #if guess == cor:
                #cum_correct += 1
            cum_correct += loss_sum

        # Return negative loss because fitness is expected
        return -cum_correct

    g = NEAT.Genome(0, subs.GetMinCPPNInputs(), 0, subs.GetMinCPPNOutputs(), False, 
                    NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.TANH, 0, params)
    pop = NEAT.Population(g, params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    current_best = None
    for generation in range((max_evaluations/batch_size)):
        for i_episode, genome in enumerate(NEAT.GetGenomeList(pop)):
            reward = objF(genome)

            genome.SetFitness(reward)

        # print('Generation: {}, max fitness: {}'.format(generation,
                            # max((x.GetFitness() for x in NEAT.GetGenomeList(pop)))))
        current_best = pickle.dumps(pop.GetBestGenome())
        testNetwork(data, n_classes, subs, current_best, num_samples * (generation + 1) * batch_size, file, seed)
        pop.Epoch()

def configure_for_training(batch_size, max_evaluations, n_classes, subs, seed, f):
    from sklearn.metrics import log_loss, accuracy_score

    def train_network(X_train, y_train, X_validate, y_validate, X_test, y_test, test_split=0, validate_split=0):
        file_start = "%d\t%d\t%d" % (seed, test_split, validate_split)
        
        evaluations_per_generation = params.PopulationSize * batch_size
        num_generations = max_evaluations/(evaluations_per_generation) + 1

        g = NEAT.Genome(0, subs.GetMinCPPNInputs(), 0, subs.GetMinCPPNOutputs(), False, 
                        NEAT.ActivationFunction.LINEAR, NEAT.ActivationFunction.TANH, 0, params)

        pop = NEAT.Population(g, params, True, 1.0, seed)
        pop.RNG.Seed(seed)

        def run_network(genome, cur_data, cur_label):
            net = NEAT.NeuralNetwork()
            genome.BuildHyperNEATPhenotype(net, subs)

            results = []
            for example in cur_data:
                net.Flush()
                net.Input(example)
                for _ in range(3):
                    net.Activate()

                result = softmax(net.Output())
                results.append(result)

            predictions = [np.argmax(result) for result in results]

            acc = accuracy_score(cur_label, predictions)
            cost = log_loss(cur_label, results)
            return acc, cost

        # Used to sample a batch with same class ratios
        from sklearn.cross_validation import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(y_train.reshape(-1), num_generations, train_size=batch_size, random_state=seed)

        for generation, (batch_index, _) in enumerate(sss):
            train_results = []
            for genome in NEAT.GetGenomeList(pop):
                acc, cost = run_network(genome, X_train[batch_index], y_train[batch_index])
                train_results.append([acc, cost])
                genome.SetFitness(-cost)

            pop.Epoch()

            validate_results = []
            for genome in NEAT.GetGenomeList(pop):
                acc, cost = run_network(genome, X_validate, y_validate)
                validate_results.append([acc, cost])
                genome.SetFitness(-cost)

            best = pop.GetBestGenome()
            acc, cost = run_network(best, X_test, y_test)

            train_evaluations = (generation + 1) * evaluations_per_generation
            
            f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "train", train_evaluations, "acc", np.max(train_results, axis=0)[0]))
            f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "val", train_evaluations, "acc", np.max(validate_results, axis=0)[0]))
            f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "test", train_evaluations, "acc", acc))
            f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "train", train_evaluations, "cost", np.min(train_results, axis=0)[1]))
            f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "val", train_evaluations, "cost", np.min(validate_results, axis=0)[1]))
            f.write("%s\t%s\t%d\t%s\t%f\n" % (file_start, "test", train_evaluations, "cost", cost))
            if generation % 100 == 0:
                f.flush()

    return train_network


def runExperiment(architecture, dataset, seed, max_evaluations, num_samples, tanh=False):
    from neuralbench.classification.dataset.create import run_validate_splits
    np.random.seed(seed)
    file_name = "hyperneat_%s_%s_%03d_e%010d_s%05d.dat" % (architecture, dataset["name"], seed, max_evaluations, num_samples)
    if tanh:
        file_name = "hyperneattanh_%s_%s_%03d_e%010d_s%05d.dat" % (architecture, dataset["name"], seed, max_evaluations, num_samples)
    f = open(file_name, 'w')
    subs = createArchitecture(architecture, dataset["name"])
    
    set_activation_functions(tanh)

    if dataset["name"] == "mnist":
        train_network = configure_for_training(num_samples, max_evaluations, 10, subs, seed, f)
        run_validate_splits(train_network, dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"], folds=10, seed=seed)
    else:
        train_network = configure_for_training(num_samples, max_evaluations, 2, subs, seed, f)
        run_validate_splits(train_network, dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"], folds=10, seed=seed)

    f.close()

