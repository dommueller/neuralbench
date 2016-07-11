import numpy as np

class Gene(object):
    def __init__(self, chromosome, parent_fitness, evaluate, test):
        self.chromosome = chromosome
        self._evaluate = evaluate
        self._test = test
        self._parent_fitness = parent_fitness
        self._fitness = None

    def __repr__(self):
        return "Fitness: %.4f - Parent Fitness: %.4f - Weight sum: %.4f\n" % (self._fitness, self._parent_fitness, np.sum(self.chromosome))

    def __str__(self):
        return "Fitness: %.4f - Parent Fitness: %.4f - Weight sum: %.4f\n" % (self._fitness, self._parent_fitness, np.sum(self.chromosome))

    @property
    def fitness(self):
        return self._fitness

    def calc_fitness(self, generation, params):
        reward = self.evaluate(generation)
        self._fitness = (1 - params.parent_fitness_decay) * self._parent_fitness + reward

        return reward

    def evaluate(self, seed):
        return self._evaluate(self.chromosome, seed)

    def test(self, seed):
        return self._test(self.chromosome, seed)

    def mutate(self, params):
        child = []
        for parent_part in self.chromosome:
            mutation_indices = np.random.uniform(0, 1, parent_part.shape) >= params.mutation_rate
            mutations = np.random.uniform(-params.mutation_power, params.mutation_power, parent_part.shape)
            mutations[mutation_indices] = 0
            child.append(parent_part + mutations)
        return Gene(np.array(child), self.fitness, self._evaluate, self._test)

    def mate(self, partner):
        parent_fitness = (self.fitness + partner.fitness) / 2.0
        child1, child2 = self._uniform_crossover(partner)
        new_gene1 = Gene(child1, parent_fitness, self._evaluate, self._test)
        new_gene2 = Gene(child2, parent_fitness, self._evaluate, self._test)
        return new_gene1, new_gene2

    def _uniform_crossover(self, partner):
        child1 = []
        child2 = []
        assert(self.chromosome.shape == partner.chromosome.shape)
        idx = np.random.uniform(0, 1, self.chromosome.shape) < 0.5
        c1 = np.array(self.chromosome, copy=True)
        c2 = np.array(partner.chromosome, copy=True)
        c1[idx] = self.chromosome[idx]
        c2[idx] = partner.chromosome[idx]

        return c1, c2

def initialize_population(build_network, evaluation_function, test_function, params):
    return [Gene(build_network().params, 0, evaluation_function, test_function) for _ in xrange(params.population_size)]

def sort_population(pop):
    sorted_pop = sorted(pop, key=lambda individual: individual.fitness, reverse=True)
    return sorted_pop

def create_new_population(params, old_pop):
    new_pop = []
    while len(new_pop) < params.population_size:
        # Calculate probabilty for each individual
        fitness_scores = np.array([individual.fitness for individual in old_pop])
        fitness_scores = fitness_scores - np.min(fitness_scores) + 1
        probs = fitness_scores / sum(fitness_scores)

        if (np.random.uniform(0, 1) < params.sexual_reproduction_proportion):
            parents = np.random.choice(len(old_pop), 2, replace=False, p=probs)
            child1, child2 = old_pop[parents[0]].mate(old_pop[parents[1]])
            new_pop.append(child1)
            new_pop.append(child2)
        else:
            parent = np.random.choice(len(old_pop), 1, replace=False, p=probs)
            child = old_pop[parent[0]].mutate(params)
            new_pop.append(child)

    params.mutation_power = params.mutation_power * params.mutation_power_decay

    # Cut offspring besides population_size
    return new_pop[:params.population_size]

def configure_leea(population, train, params, validate = None):
    elite_size = int(params.population_size * params.selection_proportion)
    elite_size = 2 if elite_size < 2 else elite_size

    def run_leea():
        generation = 0
        new_population = population
        while True:
            generation += 1

            for individual in new_population:
                individual.calc_fitness(generation, params)

            new_population = sort_population(new_population)
            current_best = new_population[0]
            new_population = create_new_population(params, new_population[:elite_size])

            if validate != None:
                validation_results = np.array([validate(individual, generation) for individual in new_population])
                current_best = new_population[np.argmax(validation_results)], copy=True

            yield generation, current_best

    return run_leea
