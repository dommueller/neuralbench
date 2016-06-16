class LeeaParams(object):
    PARENT_FITNESS_DECAY = 0.2
    STARTING_MUTATION_POWER = 0.03
    MUTATION_POWER = 0.03
    MUTATION_RATE = 0.4

    # POPULATION_SIZE = 1000
    SELECTION_PROPORTION = 0.4
    MUTATION_POWER_DECAY = 0.99
    SEXUAL_REPRODUCTION_PROPORTION = 0.5
    
    INITIAL_WEIGHT_RANGE = 4

    def __init__(self):
        self._parent_fitness_decay = None
        self._starting_mutation_power = None
        self._mutation_power_decay = None
        self._sexual_reproduction_proportion = None
        self._population_size = None
        self._mutation_power = None
        self._mutation_rate = None
        self._selection_proportion = None
        self._batch_size = None
        self._initial_weight_range = None

    @property
    def parent_fitness_decay(self):
        """Number of individuals in the population"""
        return self._parent_fitness_decay

    @parent_fitness_decay.setter
    def parent_fitness_decay(self, value):
        self._parent_fitness_decay = value

    @property
    def starting_mutation_power(self):
        """Number of individuals in the population"""
        return self._starting_mutation_power

    @starting_mutation_power.setter
    def starting_mutation_power(self, value):
        self._starting_mutation_power = value

    @property
    def mutation_power_decay(self):
        """Number of individuals in the population"""
        return self._mutation_power_decay

    @mutation_power_decay.setter
    def mutation_power_decay(self, value):
        self._mutation_power_decay = value

    @property
    def sexual_reproduction_proportion(self):
        """Number of individuals in the population"""
        return self._sexual_reproduction_proportion

    @sexual_reproduction_proportion.setter
    def sexual_reproduction_proportion(self, value):
        self._sexual_reproduction_proportion = value

    @property
    def population_size(self):
        """Number of individuals in the population"""
        return self._population_size

    @population_size.setter
    def population_size(self, value):
        self._population_size = value

    @property
    def mutation_power(self):
        """Maximum size of a weight mutation"""
        return self._mutation_power

    @mutation_power.setter
    def mutation_power(self, value):
        self._mutation_power = value
    
    @property
    def mutation_rate(self):
        """Chance that a mutation occurs"""
        return self._mutation_rate

    @mutation_rate.setter
    def mutation_rate(self, value):
        self._mutation_rate = value

    @property
    def selection_proportion(self):
        """Number of individuals that reproduce"""
        return self._selection_proportion

    @selection_proportion.setter
    def selection_proportion(self, value):
        self._selection_proportion = value
    
    @property
    def batch_size(self):
        """Number of samples each individual sees for evaluation fitness"""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def initial_weight_range(self):
        """Number of samples each individual sees for evaluation fitness"""
        return self._initial_weight_range

    @initial_weight_range.setter
    def initial_weight_range(self, value):
        self._initial_weight_range = value

    def __str__(self):
        return "%d\t%0.2f\t%0.2f\t%0.1f\t%d\t%0.1f" % (self.population_size, self.starting_mutation_power, self.mutation_rate,
                    self.selection_proportion, self.batch_size, self.initial_weight_range)




