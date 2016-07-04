import pickle
import MultiNEAT as NEAT


def configure_neat(population, train, validate = None):
    
    def run_neat():
        generation = 0
        while True:
            generation += 1
            for genome in NEAT.GetGenomeList(population):
                fitness = train(genome, generation=generation)
                genome.SetFitness(fitness)

            if validate != None:
                population.Epoch()
                for genome in NEAT.GetGenomeList(population):
                    fitness = train(genome, generation=generation)
                    genome.SetFitness(fitness)
                current_best = pickle.dumps(population.GetBestGenome())
            else:
                current_best = pickle.dumps(population.GetBestGenome())
                population.Epoch()

            yield generation, current_best

    return run_neat