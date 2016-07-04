import MultiNEAT as NEAT

# # Algorithm Parameters
# params = NEAT.Parameters()

# # Basic parameters

# # Size of population
# params.PopulationSize = 150

# # If true, this enables dynamic compatibility thresholding
# # It will keep the number of species between MinSpecies and MaxSpecies
# params.DynamicCompatibility = True

# # Min Max number of species
# params.MinSpecies = 5
# params.MaxSpecies = 10

# # Don't wipe the innovation database each generation?
# # params.InnovationsForever

# # Allow clones or nearly identical genomes to exist simultaneously in the population.
# # This is useful for non-deterministic environments,
# # as the same individual will get more than one chance to prove himself, also
# # there will be more chances the same individual to mutate in different ways.
# # The drawback is greatly increased time for reproduction. If you want to
# # search quickly, yet less efficient, leave this to true.
# # bool
# # params.AllowClones

# # GA Parameters

# # Age treshold, meaning if a species is below it, it is considered young
# params.YoungAgeTreshold = 15

# # Fitness boost multiplier for young species (1.0 means no boost)
# # Make sure it is >= 1.0 to avoid confusion
# # double
# # params.YoungAgeFitnessBoost

# # Number of generations without improvement (stagnation) allowed for a species
# params.SpeciesMaxStagnation = 15

# # Minimum jump in fitness necessary to be considered as improvement.
# # Setting this value to 0.0 makes the system to behave like regular NEAT.
# # double
# # params.StagnationDelta

# # Age threshold, meaning if a species if above it, it is considered old
# params.OldAgeTreshold = 35

# # Multiplier that penalizes old species.
# # Make sure it is < 1.0 to avoid confusion.
# # double
# # params.OldAgePenalty

# # Detect competetive coevolution stagnation
# # This kills the worst species of age >N (each X generations)
# # bool
# # params.DetectCompetetiveCoevolutionStagnation

# # Each X generation..
# # int
# # params.KillWorstSpeciesEach

# # Of age above..
# # int
# # params.KillWorstAge

# # Percent of best individuals that are allowed to reproduce. 1.0 = 100%
# params.SurvivalRate = 0.2

# # Probability for a baby to result from sexual reproduction (crossover/mating). 1.0 = 100%
# params.CrossoverRate = 0.75  # mutate only 0.25

# # If a baby results from sexual reproduction, this probability determines if mutation will
# # be performed after crossover. 1.0 = 100% (always mutate after crossover)
# params.OverallMutationRate = 0.8

# # Probability for a baby to result from inter-species mating.
# # double
# # params.InterspeciesCrossoverRate

# # Probability for a baby to result from Multipoint Crossover when mating. 1.0 = 100%
# # The default if the Average mating.
# params.MultipointCrossoverRate = 0.4

# # Performing roulette wheel selection or not?
# params.RouletteWheelSelection = False

# # For tournament selection
# # unsigned int 
# # params.TournamentSize

# # Fraction of individuals to be copied unchanged
# # double
# # params.EliteFraction

# # Phased Search parameter
# # Novelity Search parameters
# # https://github.com/peter-ch/MultiNEAT/blob/master/src/Parameters.h

# # Mutation parameters

# # Probability for a baby to be mutated with the Add-Neuron mutation.
# params.MutateAddNeuronProb = 0.3

# # Allow splitting of any recurrent links
# # bool
# # params.SplitRecurrent

# # Allow splitting of looped recurrent links
# # bool
# # params.SplitLoopedRecurrent

# # Maximum number of tries to find a link to split
# # int
# # params.NeuronTries

# # Probability for a baby to be mutated with the Add-Link mutation
# params.MutateAddLinkProb = 0.5

# # Probability for a new incoming link to be from the bias neuron;
# # double
# # params.MutateAddLinkFromBiasProb

# # Probability for a baby to be mutated with the Remove-Link mutation
# params.MutateRemLinkProb = 0.0

# # Probability for a baby that a simple neuron will be replaced with a link
# # double
# # params.MutateRemSimpleNeuronProb

# # Maximum number of tries to find 2 neurons to add/remove a link
# # unsigned int
# # params.LinkTries

# # Probability that a link mutation will be made recurrent
# params.RecurrentProb = 0.0

# # Probability that a recurrent link mutation will be looped
# # double
# # params.RecurrentLoopProb

# # Probability for a baby's weights to be mutated
# params.MutateWeightsProb = 0.90

# # Probability for a severe (shaking) weight mutation
# params.MutateWeightsSevereProb = 0.5

# # Probability for a particular gene to be mutated. 1.0 = 100%
# params.WeightMutationRate = 0.25

# # Maximum perturbation for a weight mutation
# params.WeightMutationMaxPower = 2.5

# # Maximum magnitude of a replaced weight
# params.WeightReplacementMaxPower = 5.0

# # Maximum absolute magnitude of a weight
# params.MaxWeight = 8

# # Probability for a baby's A activation function parameters to be perturbed
# # double
# # params.MutateActivationBProb

# # Probability for a baby's B activation function parameters to be perturbed
# # double
# # params.MutateActivationBProb

# # Maximum magnitude for the A parameter perturbation
# # double
# # params.ActivationAMutationMaxPower

# # Maximum magnitude for the B parameter perturbation
# # double
# # params.ActivationBMutationMaxPower

# # Maximum magnitude for time costants perturbation
# # double
# # params.TimeConstantMutationMaxPower

# # Maximum magnitude for biases perturbation
# # double
# # params.BiasMutationMaxPower

# # Activation parameter A min/max
# params.MinActivationA  = 4.9
# params.MaxActivationA  = 4.9

# # Activation parameter B min/max
# # double
# # params.MinActivationB
# # params.MaxActivationB


# # Probability for a baby that an activation function type will be changed for a single neuron
# # considered a structural mutation because of the large impact on fitness
# # double
# # params.MutateNeuronActivationTypeProb

# # Probabilities for a particular activation function appearance
# # double
# # params.ActivationFunction_SignedSigmoid_Prob
# # params.ActivationFunction_UnsignedSigmoid_Prob
# # params.ActivationFunction_Tanh_Prob
# # params.ActivationFunction_TanhCubic_Prob
# # params.ActivationFunction_SignedStep_Prob
# # params.ActivationFunction_UnsignedStep_Prob
# # params.ActivationFunction_SignedGauss_Prob
# # params.ActivationFunction_UnsignedGauss_Prob
# # params.ActivationFunction_Abs_Prob
# # params.ActivationFunction_SignedSine_Prob
# # params.ActivationFunction_UnsignedSine_Prob
# # params.ActivationFunction_Linear_Prob
# # params.ActivationFunction_Relu_Prob
# # params.ActivationFunction_Softplus_Prob

# params.ActivationFunction_SignedSigmoid_Prob = 1.0
# params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
# params.ActivationFunction_Tanh_Prob = 1.0
# params.ActivationFunction_TanhCubic_Prob = 0.0
# params.ActivationFunction_SignedStep_Prob = 1.0
# params.ActivationFunction_UnsignedStep_Prob = 0.0
# params.ActivationFunction_SignedGauss_Prob = 1.0
# params.ActivationFunction_UnsignedGauss_Prob = 0.0
# params.ActivationFunction_Abs_Prob = 1.0
# params.ActivationFunction_SignedSine_Prob = 1.0
# params.ActivationFunction_UnsignedSine_Prob = 0.0
# params.ActivationFunction_Linear_Prob = 1.0
# params.ActivationFunction_Relu_Prob = 1.0
# params.ActivationFunction_Softplus_Prob = 0.0

# # Probability for a baby's neuron time constant values to be mutated
# # double
# # params.MutateNeuronTimeConstantsProb

# # Probability for a baby's neuron bias values to be mutated
# # double params.MutateNeuronBiasesProb

# # Time constant range
# # double params.MinNeuronTimeConstant
# # double params.MaxNeuronTimeConstant

# # Bias range
# # double params.MinNeuronBias
# # double params.MaxNeuronBias

# # Speciation parameters

# # Percent of disjoint genes importance
# # double params.DisjointCoeff

# # Percent of excess genes importance
# # double params.ExcessCoeff

# # Node-specific activation parameter A difference importance
# # double params.ActivationADiffCoeff

# # Node-specific activation parameter B difference importance
# # double params.ActivationBDiffCoeff

# # Average weight difference importance
# params.WeightDiffCoeff = 4.0

# # Average time constant difference importance
# # double params.TimeConstantDiffCoeff

# # Average bias difference importance
# # double params.BiasDiffCoeff

# # Activation function type difference importance
# # double params.ActivationFunctionDiffCoeff

# # Compatibility treshold
# params.CompatTreshold = 2.0

# # Minumal value of the compatibility treshold
# # double MinCompatTreshold;

# # Modifier per generation for keeping the species stable
# # double params.CompatTresholdModifier

# # Per how many generations to change the treshold
# # unsigned int params.CompatTreshChangeInterval_Generations

# # Per how many evaluations to change the treshold
# # unsigned int params.CompatTreshChangeInterval_Evaluations

def standard_initialization():
    params = NEAT.Parameters()
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

    return params

def random_initialization(seed=None):
    import random

    if seed != None:
        random.seed(seed)
    else:
        from datetime import datetime
        random.seed(datetime.now())

    # Algorithm Parameters
    params = NEAT.Parameters()

    params.PopulationSize = random.randint(5, 1000)

    # Min Max number of species
    params.MinSpecies = min(random.randint(1, 10), params.PopulationSize)
    params.MaxSpecies = min(max(random.randint(1, 20), params.MinSpecies), params.PopulationSize)

    # GA Parameters


    # Percent of best individuals that are allowed to reproduce. 1.0 = 100%
    params.SurvivalRate = round(random.random(), 2)

    # Probability for a baby to result from sexual reproduction (crossover/mating). 1.0 = 100%
    params.CrossoverRate = round(random.random(), 2)  # mutate only 0.25

    # If a baby results from sexual reproduction, this probability determines if mutation will
    # be performed after crossover. 1.0 = 100% (always mutate after crossover)
    params.OverallMutationRate = round(random.random(), 2)

    # Probability for a baby to result from Multipoint Crossover when mating. 1.0 = 100%
    # The default if the Average mating.
    params.MultipointCrossoverRate = round(random.random(), 2)

    # Mutation parameters

    # Probability for a baby to be mutated with the Add-Neuron mutation.
    params.MutateAddNeuronProb = round(random.random(), 2)

    # Probability for a baby to be mutated with the Add-Link mutation
    params.MutateAddLinkProb = round(random.random(), 2)

    # Probability that a link mutation will be made recurrent
    params.RecurrentProb = round(random.random(), 2)

    # Probability for a baby's weights to be mutated
    params.MutateWeightsProb = round(random.random(), 2)

    # Probability for a particular gene to be mutated. 1.0 = 100%
    params.WeightMutationRate = round(random.random(), 2)

    # Probabilities for a particular activation function appearance
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

    return params

def write_params_header():
    return "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("populationsize", "minspecies", "maxspecies", "survivalrate", "crossoverrate", "overallmutationrate", "multipointcrossoverrate", "mutateaddneuronprob", "mutateaddlinkprob", "recurrentprob", "mutateweightsprob", "weightmutationrate")

def write_params(params):
    return "%d\t%d\t%d\t%.02f\t%.02f\t%.02f\t%.02f\t%.02f\t%.02f\t%.02f\t%.02f\t%.02f" % (params.PopulationSize, params.MinSpecies, params.MaxSpecies, params.SurvivalRate, params.CrossoverRate, params.OverallMutationRate, params.MultipointCrossoverRate, params.MutateAddNeuronProb, params.MutateAddLinkProb, params.RecurrentProb, params.MutateWeightsProb, params.WeightMutationRate)


