import logging
import math

import numpy as np

from splendor.SplendorGame import getGameEnded, getNextState, getValidMoves, getCanonicalForm
from numba import njit

EPS = 1e-8
NAN = -42.
MINFLOAT = float('-inf')

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, dirichlet_noise=False):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.dirichlet_noise = dirichlet_noise

        # Contains tuple of Es, Vs, Ps, Ns, Qsa, Nsa
        #       Vs stores game.getGameEnded ended for board s
        #       Es stores game.getValidMoves for board s
        #       Ps stores initial policy (returned by neural net)    
        #       Ns stores #times board s was visited
        #       Qsa stores Q values for s,a (as defined in the paper)
        #       Nsa stores #times edge s,a was visited
        self.nodes_data = {} # stores data for each nodes in a single dictionary
        self.Qsa_default = np.full (self.game.getActionSize(), NAN, dtype=np.float64)
        self.Nsa_default = np.zeros(self.game.getActionSize()     , dtype=np.int64)

        self.rng = np.random.default_rng()

    def getActionProb(self, canonicalBoard, temp=1, force_full_search=False):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        if self.game.getRound(canonicalBoard) == 0:
            # first move, cleaning tree
            self.nodes_data = {}

        is_full_search = force_full_search or (self.rng.random() < self.args.prob_fullMCTS)
        nb_MCTS_sims = self.args.numMCTSSims if is_full_search else self.args.numMCTSSims // self.args.ratio_fullMCTS
        for i in range(nb_MCTS_sims):
            dir_noise = (i == 0 and self.dirichlet_noise and is_full_search)
            self.search(canonicalBoard, dirichlet_noise=dir_noise)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.nodes_data[s][5][a] for a in range(self.game.getActionSize())]

        # Compute kl-divergence on probs vs self.Ps[s]
        probs = np.array(counts)
        probs = probs / probs.sum()
        surprise = (np.log(probs+EPS) - np.log(self.nodes_data[s][2]+EPS)).dot(probs).item()

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs, surprise, is_full_search

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs, surprise, is_full_search

    def search(self, canonicalBoard, dirichlet_noise=False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)
        Es, Vs, Ps, Ns, Qsa, Nsa = self.nodes_data.get(s, (None, )*6)

        if Es is None:
            Es = getGameEnded(self.game.board, canonicalBoard, 1)
            if Es != 0:
                # terminal node
                self.nodes_data[s] = (Es, Vs, Ps, Ns, Qsa, Nsa)
                return -Es
        elif Es != 0:
            # terminal node
            return -Es

        if Ps is None:
            Vs = getValidMoves(self.game.board, canonicalBoard, 1)
            # leaf node
            Ps, v = self.nnet.predict(canonicalBoard, Vs)
            if dirichlet_noise:
                self.applyDirNoise(Ps, Vs)
            normalise(Ps)

            Ns, Qsa, Nsa = 0, self.Qsa_default.copy(), self.Nsa_default.copy()
            self.nodes_data[s] = (Es, Vs, Ps, Ns, Qsa, Nsa)
            return -v

        if dirichlet_noise:
            self.applyDirNoise(Ps, Vs)
            normalise(Ps)

        # pick the action with the highest upper confidence bound
        # get next state and get canonical version of it
        a, next_s = get_next_best_action_and_canonical_state(
            Es, Vs, Ps, Ns, Qsa, Nsa,
            self.args.cpuct,
            self.game.board,
            canonicalBoard
        )

        v = self.search(next_s)

        Qsa[a] = (Nsa[a] * Qsa[a] + v) / (Nsa[a] + 1) # if Qsa[a] is NAN, then Nsa is zero
        Nsa[a] += 1
        Ns += 1

        self.nodes_data[s] = (Es, Vs, Ps, Ns, Qsa, Nsa)
        return -v


    def applyDirNoise(self, Ps, Vs):
        dir_values = self.rng.dirichlet([self.args.dirichletAlpha] * np.count_nonzero(Vs))
        dir_idx = 0
        for idx in range(len(Ps)):
            if Vs[idx]:
               Ps[idx] = (0.75 * Ps[idx]) + (0.25 * dir_values[dir_idx])
               dir_idx += 1


# pick the action with the highest upper confidence bound
@njit(cache=True, fastmath=True)
def pick_highest_UCB(Es, Vs, Ps, Ns, Qsa, Nsa, cpuct):
    cur_best = MINFLOAT
    best_act = -1

    for a, valid in enumerate(Vs):
        if valid:
            if Qsa[a] != NAN:
                u = Qsa[a] + cpuct * Ps[a] * math.sqrt(Ns) / (1 + Nsa[a])
            else:
                u = cpuct * Ps[a] * math.sqrt(Ns + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best, best_act = u, a

    return best_act


@njit(cache=True, fastmath=True)
def get_next_best_action_and_canonical_state(Es, Vs, Ps, Ns, Qsa, Nsa, cpuct, gameboard, canonicalBoard):
    a = pick_highest_UCB(Es, Vs, Ps, Ns, Qsa, Nsa, cpuct)

    next_s, next_player = getNextState(gameboard, canonicalBoard, 1, a)
    next_s = getCanonicalForm(gameboard, next_s, next_player)

    return a, next_s

@njit(cache=True, fastmath=True)
def normalise(vector):
    sum_vector = np.sum(vector)
    vector /= sum_vector