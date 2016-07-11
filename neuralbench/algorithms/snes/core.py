from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import ReluLayer
from pybrain.structure.modules import LinearLayer
from pybrain.structure.modules import LSTMLayer
from pybrain.optimization import SNES

print __name__

def configure_snes(objF, start_params, minimize=False):
    l = SNES(objF, start_params, verbose=False)
    l.minimize = minimize

    def run_snes():
        global generation
        generation = 0

        while True:
            generation += 1
            result = l.learn(additionalLearningSteps=1)
            current_best = result[0]

            yield current_best

    return run_snes, l.batchSize