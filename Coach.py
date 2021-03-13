import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import time

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, self.nnet.args)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args, dirichlet_noise=(self.args.dirichletAlpha>0))
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

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
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi, surprise, is_full_search = self.mcts.getActionProb(canonicalBoard, temp=temp)
            if is_full_search:
                valids = self.game.getValidMoves(canonicalBoard, 1)
                sym = self.game.getSymmetries(canonicalBoard, pi, valids)
                for b, p, v in sym:
                    trainExamples.append([b, self.curPlayer, p, v, surprise])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)
            if r != 0:
                final_scores_diff = self.game.getScore(board, self.curPlayer) - self.game.getScore(board, -self.curPlayer)
                return [(
                    x[0],                                # board
                    x[2],                                # policy
                    r if x[1] == self.curPlayer else -r, # winner
                    final_scores_diff if x[1] == self.curPlayer else -final_scores_diff, # score difference
                    x[3],                                # valids
                    x[4],                                # surprise
                ) for x in trainExamples]

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

                for _ in tqdm(range(self.args.numEps), desc="Self Play", ncols=100):
                    self.mcts = MCTS(self.game, self.nnet, self.args, dirichlet_noise=(self.args.dirichletAlpha>0))  # reset search tree
                    iterationTrainExamples += self.executeEpisode()
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
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            if i % 5 == 0:
                self.saveTrainExamples(i - 1) # HUGE PEAK, MEMORY CONSUMPTION TOO HIGH
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pt')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pt')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            # log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(nmcts.getActionProb(x, temp=0, force_full_search=True)[0]),
                          lambda x: np.argmax(pmcts.getActionProb(x, temp=0, force_full_search=True)[0]), self.game)
            nwins, pwins, draws = arena.playGames(self.args.arenaCompare)

            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info(f'new vs previous: {nwins}-{pwins}  ({draws} draws) --> REJECTED')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pt')
            else:
                log.info(f'new vs previous: {nwins}-{pwins}  ({draws} draws) --> ACCEPTED')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pt')

            if self.args.timeIters > 0:
                if time.time() - start_time > self.args.timeIters*3600:
                    log.info(f'Above timelimit, stopping here after {i} iterations')
                    break

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pt'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, "checkpoint.examples")
        with open(filename, "wb") as f:
            Pickler(f).dump(self.trainExamplesHistory)

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
            self.trainExamplesHistory = Unpickler(f).load()
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