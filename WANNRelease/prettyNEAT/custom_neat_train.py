import os
import time
import argparse
import numpy as np

np.set_printoptions(precision=2, linewidth=160)

# neat_src and domain should be available in your Colab environment
from neat_src import *  # NEAT
from domain import *    # Task environments


# -- Run NEAT ------------------------------------------------------------ -- #
def run_neat():
    """Main NEAT optimization script (Sequential version)"""
    global fileName, hyp
    data = DataGatherer(fileName, hyp)
    neat = Neat(hyp)
    task = GymTask(games[hyp['task']], nReps=hyp['alg_nReps'])

    for gen in range(hyp['maxGen']):
        pop = neat.ask()  # Get newly evolved individuals
        reward = batchEval(pop, task)  # Evaluate sequentially
        neat.tell(reward)  # Send fitness back to NEAT

        data = gatherData(data, neat, gen, hyp)
        print(gen, '\t - \t', data.display())

    # Save final population
    data = gatherData(data, neat, gen, hyp, savePop=True)
    data.save()
    data.savePop(neat.pop, fileName)


def batchEval(pop, task):
    """Sequential evaluation of the population"""
    return np.array([task.getFitness(ind.wMat, ind.aVec) for ind in pop])


def gatherData(data, neat, gen, hyp, savePop=False):
    data.gatherData(neat.pop, neat.species)
    if gen % hyp['save_mod'] == 0:
        data = checkBest(data, neat)
        data.save(gen)

    if savePop:  # Save population for inspection
        pref = 'log/' + fileName
        import pickle
        with open(pref + '_pop.obj', 'wb') as fp:
            pickle.dump(neat.pop, fp)

    return data


def checkBest(data, neat):
    """Checks if new best individual is truly better over many trials"""
    global hyp
    if data.newBest:
        task = GymTask(games[hyp['task']], nReps=hyp['alg_nReps'])
        bestReps = hyp['bestReps']
        rep = [data.best[-1]] * bestReps
        fitVector = np.array([task.getFitness(ind.wMat, ind.aVec) for ind in rep])
        trueFit = np.mean(fitVector)
        if trueFit > data.best[-2].fitness:
            data.best[-1].fitness = trueFit
            data.fit_top[-1] = trueFit
            data.bestFitVec = fitVector
        else:
            prev = hyp['save_mod']
            data.best[-prev:] = data.best[-prev]
            data.fit_top[-prev:] = data.fit_top[-prev]
            data.newBest = False
    return data


# -- Input Parsing ------------------------------------------------------- -- #

def main(args):
    """Loads hyperparameters and starts NEAT run"""
    global fileName, hyp
    fileName = args.outPrefix
    hyp_default = args.default
    hyp_adjust = args.hyperparam
    # print(f"fileName :{fileName}")
    # print(f"hyp_default :{hyp_default}")
    # print(f"hyp_adjust :{hyp_adjust}")
    hyp = loadHyp(pFileName=hyp_default)
    updateHyp(hyp, hyp_adjust)

    run_neat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evolve NEAT networks')
    parser.add_argument('-d', '--default', type=str, default='p/default_neat.json',
                        help='default hyperparameter file')
    parser.add_argument('-p', '--hyperparam', type=str, default=None,
                        help='additional hyperparameter overrides')
    parser.add_argument('-o', '--outPrefix', type=str, default='test',
                        help='prefix for output files')
    args = parser.parse_args()
    main(args)
