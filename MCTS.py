import logging
import math
import numpy as np
import gc
from random import randrange
from copy import deepcopy

from numba import njit

EPS = 1e-8
NAN = -42.
k = 0.5
MINFLOAT = float('-inf')

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, dirichlet_noise=False, batch_info=None):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.dirichlet_noise = dirichlet_noise

        # Contains tuple of Es, Vs, Ps, Ns, Qsa, Nsa
        #       Es stores game.getGameEnded ended for board s
        #       Vs stores game.getValidMoves for board s
        #       Ps stores initial policy (returned by neural net)    
        #       Ns stores #times board s was visited
        #       Qsa stores Q values for s,a (as defined in the paper)
        #       Nsa stores #times edge s,a was visited
        #       r stores round number
        #       Qs stores Q value for s
        self.nodes_data = {} # stores data for each nodes in a single dictionary
        self.Qsa_default = np.full (self.game.getActionSize(), NAN, dtype=np.float64)
        self.Nsa_default = np.zeros(self.game.getActionSize()     , dtype=np.int64)

        self.rng = np.random.default_rng()
        self.step = 0
        self.last_cleaning = 0
        self.batch_info = batch_info
        self.random_seed = 0

    def getActionProb(self, canonicalBoard, temp=1, force_full_search=False):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        is_full_search = force_full_search or (self.rng.random() < self.args.prob_fullMCTS)
        nb_MCTS_sims = self.args.numMCTSSims if is_full_search else self.args.numMCTSSims // self.args.ratio_fullMCTS
        forced_playouts = (is_full_search and self.args.forced_playouts)

        nb_universes = abs(self.args.universes)
        random_seeds = [randrange(1, 2**16) for _ in range(nb_universes)] if self.args.universes >= 0 else [1984, 31415, 3108, 1411][:nb_universes]
        for self.step in range(nb_MCTS_sims):
            self.random_seed = random_seeds[self.step % nb_universes] if self.args.universes != 0 else -1
            dir_noise = (self.step == 0 and is_full_search and self.dirichlet_noise)
            self.search(canonicalBoard, dirichlet_noise=dir_noise, forced_playouts=forced_playouts)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.nodes_data[s][5][a] for a in range(self.game.getActionSize())] # Nsa

        # Compute Q at root node
        q_player0 = self.nodes_data[s][7]
        q = [q_player0 if n == 0 else -q_player0/(self.game.num_players-1) for n in range(self.game.num_players)]

        # Policy target pruning
        if forced_playouts:
            best_count = max(counts)
            Psas   = [self.nodes_data[s][2][a] for a in range(self.game.getActionSize())] # Ps[a]
            adjusted_counts = [Nsa-int(math.sqrt(k*Psa*nb_MCTS_sims)) if Nsa != best_count else Nsa for (Nsa, Psa) in zip(counts, Psas)]
            adjusted_counts = [c if c > 1 else 0 for c in adjusted_counts]
            counts = adjusted_counts

        probs = np.array(counts)
        probs = probs / probs.sum()

        # Clean search tree from very old moves = less memory footprint and less keys to search into
        if not self.args.no_mem_optim:
            r = self.game.getRound(canonicalBoard)
            if r > self.last_cleaning + 20:
                for node in [n for n in self.nodes_data.keys() if self.nodes_data[n][6] < r-5]:
                    del self.nodes_data[node]
                self.last_cleaning = r

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs, q, is_full_search

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs, q, is_full_search

    def search(self, canonicalBoard, dirichlet_noise=False, forced_playouts=False):
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
        Es, Vs, Ps, Ns, Qsa, Nsa, r, Qs = self.nodes_data.get(s, (None, )*8)
        if r is None:
            r = self.game.getRound(canonicalBoard)

        if Es is None:
            Es = self.game.getGameEnded(canonicalBoard, 0)
            if Es.any():
                # terminal node
                self.nodes_data[s] = (Es, Vs, Ps, Ns, Qsa, Nsa, r, Qs)
                return Es
        elif Es.any():
            # terminal node
            return Es

        if Ps is None:
            # First time that we explore state s
            Vs = self.game.getValidMoves(canonicalBoard, 0)
            if self.batch_info is None:
                Ps, v = self.nnet.predict(canonicalBoard, Vs)
            else:
                Ps, v = self.nnet.predict_client(canonicalBoard, Vs, self.batch_info)
            if dirichlet_noise:
                Ps = softmax(Ps, self.args.temperature[0])
                self.applyDirNoise(Ps, Vs)
            normalise(Ps)

            Ns, Qsa, Nsa = 0, self.Qsa_default.copy(), self.Nsa_default.copy()
            self.nodes_data[s] = (Es, Vs, Ps, Ns, Qsa, Nsa, r, v[0])
            return v

        if dirichlet_noise:
            # We already visited this node, adding dirichlet noise this time
            Ps = softmax(Ps, self.args.temperature[0])
            self.applyDirNoise(Ps, Vs)
            normalise(Ps)

        # pick the action with the highest upper confidence bound
        # get next state and get canonical version of it
        a, next_s, next_player = get_next_best_action_and_canonical_state(
            Es, Vs, Ps, Ns, Qsa, Nsa, Qs,
            self.args.cpuct,
            self.game.board,
            canonicalBoard,
            forced_playouts,
            self.step,
            self.args.fpu,
            self.random_seed,
        )

        v = self.search(next_s)
        v = np_roll(v, next_player)

        Qsa[a] = (Nsa[a] * Qsa[a] + v[0]) / (Nsa[a] + 1) # if Qsa[a] is NAN, then Nsa is zero
        Qs = ((Ns+1) * Qs + v[0]) / (Ns+2) # Qs can't be None here
        Nsa[a] += 1
        Ns += 1

        self.nodes_data[s] = (Es, Vs, Ps, Ns, Qsa, Nsa, r, Qs)
        return v


    def applyDirNoise(self, Ps, Vs):
        if self.args.dirichletAlpha > 0:
            dir_values = self.rng.dirichlet([self.args.dirichletAlpha] * np.count_nonzero(Vs))
        elif self.args.dirichletAlpha < 0:
            # Automatic value
            dir_values = self.rng.dirichlet([10 / np.count_nonzero(Vs)] * np.count_nonzero(Vs))
        dir_idx = 0
        for idx in range(len(Ps)):
            if Vs[idx]:
               Ps[idx] = (0.75 * Ps[idx]) + (0.25 * dir_values[dir_idx])
               dir_idx += 1

    @staticmethod
    def reset_all_search_trees():
        for obj in [o for o in gc.get_objects() if type(o) is MCTS]: # dirtier than isinstance, but that would trigger a pytorch warning
            obj.nodes_data = {}
            obj.last_cleaning = 0
        
@njit(cache=True, fastmath=True, nogil=True)
def np_roll(arr, n):
    return np.roll(arr, n)

# pick the action with the highest upper confidence bound
@njit(cache=True, fastmath=True, nogil=True)
def pick_highest_UCB(Es, Vs, Ps, Ns, Qsa, Nsa, Qs, cpuct, forced_playouts, n_iter, fpu):
    cur_best = MINFLOAT
    best_act = -1
    fpu_init = Qs-fpu if fpu > 0 else fpu

    for a, valid in enumerate(Vs):
        if valid:
            if forced_playouts:
                if Nsa[a] < int(math.sqrt(k * Ps[a] * n_iter)): # Nsa is zero when not set
                    return a

            if Qsa[a] != NAN:
                u = Qsa[a] + cpuct * Ps[a] * math.sqrt(Ns) / (1 + Nsa[a])
            else:
                u = fpu_init + cpuct * Ps[a] * math.sqrt(Ns + EPS)

            if u > cur_best:
                cur_best, best_act = u, a

    return best_act


@njit(fastmath=True, nogil=True) # no cache because it relies on jitclass which isn't compatible with cache
def get_next_best_action_and_canonical_state(Es, Vs, Ps, Ns, Qsa, Nsa, Qs, cpuct, gameboard, canonicalBoard, forced_playouts, n_iter, fpu, random_seed):
    a = pick_highest_UCB(Es, Vs, Ps, Ns, Qsa, Nsa, Qs, cpuct, forced_playouts, n_iter, fpu)

    # Do action 'a'
    gameboard.copy_state(canonicalBoard, True)
    next_player = gameboard.make_move(a, 0, deterministic=random_seed)
    # next_s = gameboard.get_state()

    # Get canonical form
    if next_player != 0:
        # gameboard.copy_state(next_s, True)
        gameboard.swap_players(next_player)
    next_s = gameboard.get_state()

    return a, next_s, next_player

@njit(cache=True, fastmath=True, nogil=True)
def normalise(vector):
    sum_vector = np.sum(vector)
    vector /= sum_vector

@njit(cache=True, fastmath=True, nogil=True)
def softmax(Ps, softmax_temp):
    if softmax_temp == 1.:
        return Ps
    result = Ps ** (1. / softmax_temp)
    normalise(result)
    return result.astype(np.float32)
