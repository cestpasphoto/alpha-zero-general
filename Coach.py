import logging
import os
import sys
from collections import deque
import pickle
import zlib
from random import shuffle
import time

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS, random_pick

log = logging.getLogger(__name__)

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        print(f'Nb of coef: {sum(p.numel() for p in self.nnet.nnet.parameters())}')
        self.pnet = self.nnet.__class__(self.game, self.nnet.args)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args, dirichlet_noise=(self.args.dirichletAlpha>0))
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = (args.load_model and args.forget)  # can be overriden in loadTrainExamples()
        self.nnet.force_long_training = (args.load_model and args.forget)

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 0
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)

            pi, surprise, is_full_search = self.mcts.getActionProb(canonicalBoard, temp=self.args.temperature)
            if is_full_search:
                valids = self.game.getValidMoves(canonicalBoard, 0)
                sym = self.game.getSymmetries(canonicalBoard, pi, valids)
                for b, p, v in sym:
                    trainExamples.append([b, self.curPlayer, p, v, surprise])

            final_temp = 0.2/self.args.temperature
            early_temp = 5*final_temp
            temp = final_temp + (early_temp-final_temp)*(0.5**(episodeStep/self.args.tempThreshold))
            # print(f'{self.args.temperature=} {episodeStep=} temp={round(temp,2)}')
            action = random_pick(pi, temp)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)
            if r.any():
                final_scores = [self.game.getScore(board, p) for p in range(self.game.num_players)]
                trainExamples = [(
                    x[0],                                # board
                    x[2],                                # policy
                    np.roll(r, -x[1]),                   # winner
                    np.roll([f-final_scores[x[1]] for f in final_scores], -x[1]), # score difference
                    x[3],                                # valids
                    x[4],                                # surprise
                ) for x in trainExamples]

                return trainExamples if self.args.no_compression else [zlib.compress(pickle.dumps(x), level=1) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        start_time = time.time()

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play", ncols=120):
                    iterationTrainExamples += self.executeEpisode()
                    MCTS.reset_all_search_trees()
                    if len(iterationTrainExamples) == self.args.maxlenOfQueue:
                        log.warning(f'saturation of elements in iterationTrainExamples, think about decreasing numEps or increasing maxlenOfQueue')
                        break

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if self.args.profile:
                return

            if len(self.trainExamplesHistory) > self.args.numItersHistory:
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            self.saveTrainExamples()
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pt', additional_keys=vars(self.args))
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pt')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            # log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: random_pick((nmcts.getActionProb(x, force_full_search=True)[0]), temperature=0.2),
                          lambda x: random_pick((pmcts.getActionProb(x, force_full_search=True)[0]), temperature=0.2),
                          self.game)
            nwins, pwins, draws = arena.playGames(self.args.arenaCompare)

            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info(f'new vs previous: {nwins}-{pwins}  ({draws} draws) --> REJECTED')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pt')
            else:
                log.info(f'new vs previous: {nwins}-{pwins}  ({draws} draws) --> ACCEPTED')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i), additional_keys=vars(self.args))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pt', additional_keys=vars(self.args))

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pt'

    def saveTrainExamples(self):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, "checkpoint.examples")
        with open(filename, "wb") as f:
            pickle.dump(self.trainExamplesHistory, f)

    def loadTrainExamples(self):
        modelFile = self.args.load_folder_file
        examplesFile = os.path.dirname(modelFile) + "/checkpoint.examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
            return
    
        log.info("File with trainExamples found. Loading it...")
        with open(examplesFile, "rb") as f:
            self.trainExamplesHistory = pickle.load(f)
        
        # Harmonize compression use in loaded examples
        if type(self.trainExamplesHistory[0][0]) is tuple and not self.args.no_compression:
            for i in range(len(self.trainExamplesHistory)):
                for j in range(len(self.trainExamplesHistory[i])):                    
                    self.trainExamplesHistory[i][j] = zlib.compress(pickle.dumps(self.trainExamplesHistory[i][j]), level=1)
        elif type(self.trainExamplesHistory[0][0]) is not tuple and self.args.no_compression:
            for i in range(len(self.trainExamplesHistory)): 
                for j in range(len(self.trainExamplesHistory[i])):
                    self.trainExamplesHistory[i][j] = pickle.loads(zlib.decompress(self.trainExamplesHistory[i][j]))
        log.info('Loading done!')

        # cleaning
        if len(self.trainExamplesHistory) > self.args.numItersHistory:
            self.trainExamplesHistory = self.trainExamplesHistory[-self.args.numItersHistory:]
            log.info('Reduced history in loaded examples')
        for history in self.trainExamplesHistory:
            if len(history) > self.args.maxlenOfQueue:
                for _ in range(len(history), self.args.maxlenOfQueue, -1):
                    history.pop()
                log.info('Reduced nb of items in one history of loaded examples')
