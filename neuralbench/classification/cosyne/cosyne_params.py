class CosyneParams(object):
    def __init__(self):
        self._population_size = None
        self._mutation_power = None
        self._mutation_rate = None
        self._selection_proportion = None
        self._batch_size = None
        self._initial_weight_range = None

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
        return "%d %0.2f %0.2f %0.1f %d %0.1f" % (self.population_size, self.mutation_power, self.mutation_rate,
                    self.selection_proportion, self.batch_size, self.initial_weight_range)




