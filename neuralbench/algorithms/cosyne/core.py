import numpy as np

def initialize_population(build_network, params):
    return np.array([build_network().params for _ in xrange(params.population_size)])

def uniform_crossover(parent1, parent2):
    assert(parent1.shape == parent2.shape)
    idx = np.random.uniform(0, 1, parent1.shape) < 0.5
    child1 = parent1
    child2 = parent2
    child1[idx] = parent2[idx]
    child2[idx] = parent1[idx]

    return child1, child2

def uniform_mutation(parent, params):
    mutation_indices = np.random.uniform(0, 1, parent.shape) >= params.mutation_rate
    mutations = np.random.uniform(-params.mutation_power, params.mutation_power, parent.shape)
    mutations[mutation_indices] = 0
    return parent + mutations

# Shuffles weights in one subpoplation / shuffles columns
def permute_inplace(population):
    transposed = population.T
    map(np.random.shuffle, transposed)

def create_new_generation(old_pop, results, params):
    top_idx = int(len(old_pop) * params.selection_proportion)
    top_idx = 2 if top_idx < 2 else top_idx
    elite = np.array(old_pop[0:top_idx], copy=True)
    offspring = []
    needed_offsprings = len(old_pop) - top_idx
    for _ in xrange(needed_offsprings/2 + 1):
        # parents = np.random.choice(len(elite), 2, replace=False, p=softmax(-np.array(results[0:top_idx])))
        # Substract max, so that lowest will have highest prob
        # new_results = np.array(results[0:top_idx]) - (results[-1] + 1)
        fitness_scores = results[0:top_idx]
        fitness_scores = fitness_scores - np.min(fitness_scores) + 1
        probs = fitness_scores / float(sum(fitness_scores))
        parents = np.random.choice(len(elite), 2, replace=False, p=probs)
        child1, child2 = uniform_crossover(elite[parents[0]], elite[parents[1]])
        child1 = uniform_mutation(child1, params)
        child2 = uniform_mutation(child2, params)
        offspring.append(child1)
        offspring.append(child2)

    permute_inplace(elite)
    new_pop = np.append(elite, offspring, axis=0)
    new_pop = new_pop[:len(old_pop)]
    assert(old_pop.shape == new_pop.shape)

    return new_pop


def configure_cosyne(population, train, params, validate = None):
    def run_cosyne():
        generation = 0
        new_population = population
        while True:
            generation += 1

            results = np.array([train(individium, generation) for individium in new_population])
            sort_idx = np.argsort(results)[::-1]
            sorted_pop = new_population[sort_idx]
            sorted_results = results[sort_idx]
            current_best = np.array(sorted_pop[0], copy=True)
            new_population = create_new_generation(sorted_pop, sorted_results, params)

            if validate != None:
                validation_results = np.array([validate(individium, generation) for individium in new_population])
                current_best = np.array(new_population[np.argmax(validation_results)], copy=True)
            
            yield generation, current_best

    return run_cosyne

