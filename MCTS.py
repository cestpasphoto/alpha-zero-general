import logging
import math
import gc

import numpy as np
from numba import njit

from Stochastic import hashed_draw

EPS = 1e-8
NAN = -42.
MINFLOAT = float('-inf')
# Fixed chance seeds used inside the tree, cycled over simulations (args.universes <= 8).
# They never change between moves nor games: universes=1 plans every search against
# the same chance stream. Real games use random_seed=0, i.e. true randomness.
magic_seeds = [31416, 1, 14142, 42, 27183, 2, 16180, 7]

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, dirichlet_noise=False, batch_info=None, is_selfplay=False):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.dirichlet_noise = dirichlet_noise
        self.is_selfplay = is_selfplay

        # One entry per board s: (Es, Vs, Ps, [Ns, Qs], Qsa, Nsa, r)
        #   Es  game.getGameEnded(s)          Vs   game.getValidMoves(s)
        #   Ps  prior returned by the net     Ns   visit count of s
        #   Qs  per-player value vector of s  Qsa / Nsa  value / visits of edge (s,a)
        #   r   round number, used to prune old nodes
        self.nodes_data = {}
        self.Qsa_default = np.full (self.game.getActionSize(), NAN, dtype=np.float32)
        self.Nsa_default = np.zeros(self.game.getActionSize()     , dtype=np.int16)

        self.rng = np.random.default_rng()
        self.step = 0
        self.last_cleaning = 0
        self.batch_info = batch_info
        self.random_seed = -1

        # Hidden-information games expose Game.getObservation / Game.sampleWorld.
        # The net is then always queried on the observation of the player to move,
        # which is what it is trained on.
        self.hidden_info = hasattr(self.game, 'getObservation')
        # A game may ask for a fresh chance stream at every simulation: the tree
        # then averages over dice / draws instead of planning against one known
        # future, and a universe only fixes the invented hidden information.
        # Only worth it when chance events have few outcomes, otherwise the tree
        # fragments into many rarely visited nodes.
        self.chance_per_sim = bool(getattr(self.game, 'chance_per_sim', False))
        self._fp_warned = False

    def getActionProb(self, canonicalBoard, temp=1, force_full_search=False):
        """
        Performs numMCTSSims simulations of MCTS starting from canonicalBoard.

        Returns:
            probs: policy vector, proportional to Nsa[(s,a)]**(1./temp)
            q: per-player value vector at the root
            is_full_search: False for a Playout Cap Randomization fast search
        """
        is_full_search = force_full_search or (self.rng.random() < self.args.prob_fullMCTS)
        nb_MCTS_sims = self.args.numMCTSSims if is_full_search else self.args.numMCTSSims // self.args.ratio_fullMCTS
        forced_playouts = (is_full_search and self.args.forced_playouts)
        # Self-play full searches start from a fresh tree so that forced playouts,
        # policy target pruning and Dirichlet noise act on uncontaminated counts.
        if is_full_search and self.is_selfplay:
            self.nodes_data = {}
            self.last_cleaning = 0

        # Hidden information: one invented world per universe, dealt once from the
        # observation, then searched as a perfect-information game. The tree key is
        # the board, so two universes inventing the same world share one root node.
        universe_roots = {}
        root_keys = set()
        if self.hidden_info:
            obs = self.game.getObservation(canonicalBoard, 0)
        chance_base = int(self.rng.integers(1, 2147483647)) if self.chance_per_sim else 0

        for self.step in range(nb_MCTS_sims):
            world_seed = magic_seeds[self.step % self.args.universes] if self.args.universes > 0 else -1
            # never 0: 0 means true randomness in the game logic
            self.random_seed = (1 + hashed_draw(chance_base, self.step, 2147483646)) if self.chance_per_sim else world_seed
            if self.hidden_info:
                is_new_root = False
                if world_seed not in universe_roots:
                    universe_roots[world_seed] = self.game.sampleWorld(obs, world_seed)
                    rk = self.game.stringRepresentation(universe_roots[world_seed])
                    is_new_root = rk not in root_keys   # a shared root is noised once only
                    root_keys.add(rk)
                root_board = universe_roots[world_seed]
                dir_noise = (is_new_root and is_full_search and self.dirichlet_noise)
            else:
                root_board = canonicalBoard
                dir_noise = (self.step == 0 and is_full_search and self.dirichlet_noise)
            self.search(root_board, dirichlet_noise=dir_noise, forced_playouts=forced_playouts and not self.hidden_info, is_root=True)

        action_size = self.game.getActionSize()
        if self.hidden_info:
            # Aggregate visits and Q over the distinct roots. A root shared by m
            # universes already holds their m x sims/u visits, so it is counted
            # once. A root that is terminal in its invented world contributes nothing.
            counts = [0] * action_size
            q_acc, n_roots = None, 0
            counted = set()
            for root_board in universe_roots.values():
                rs = self.game.stringRepresentation(root_board)
                node = self.nodes_data.get(rs)
                if node is None or node[3] is None:
                    continue
                if rs not in counted:
                    counted.add(rs)
                    for a in range(action_size):
                        counts[a] += int(node[5][a])
                q_this = node[3][1]
                q_acc = q_this.copy() if q_acc is None else q_acc + q_this
                n_roots += 1
            if n_roots > 0:
                q = list(q_acc / n_roots)
            else:
                # every invented root is terminal: fall back on the net's own view
                valids_fallback = self.game.getValidMoves(canonicalBoard, 0)
                _, v_fb = self.nnet.predict(obs, valids_fallback)
                counts = [int(valids_fallback[a]) for a in range(action_size)]
                q = list(v_fb)
            valid_moves_mask = self.game.getValidMoves(canonicalBoard, 0)  # legality reads public info only
            if forced_playouts and not self._fp_warned:
                log.warning('forced_playouts/PTP are not applied for hidden-info games (no single root Ps/Qsa '
                            'to prune against multiple determinized roots); ignoring -F for this game.')
                self._fp_warned = True
        else:
            s = self.game.stringRepresentation(canonicalBoard)
            counts = [int(n) for n in self.nodes_data[s][5]] # Nsa

            # Per-player Q measured directly from backups (no zero-sum assumption)
            q = list(self.nodes_data[s][3][1])

            # Policy target pruning: subtract up to n_forced playouts from each non-best child,
            # but stop before PUCT(a) would reach PUCT(best) (holding final utilities constant).
            if forced_playouts:
                Ps_root  = self.nodes_data[s][2]
                Qsa_root = self.nodes_data[s][4]
                S = float(sum(counts))
                best_a = int(np.argmax(counts))
                best_count = counts[best_a]
                puct_best = Qsa_root[best_a] + self.args.cpuct * Ps_root[best_a] * math.sqrt(S) / (1 + best_count)
                adjusted_counts = list(counts)
                for a in range(len(counts)):
                    n = counts[a]
                    if n == 0 or a == best_a:
                        continue
                    n_forced = int(math.sqrt(self.args.forced_playouts_k * Ps_root[a] * S))
                    gap = puct_best - Qsa_root[a]
                    if gap <= 0:
                        continue   # already at least as urgent as best: subtract nothing
                    n_min = math.ceil(self.args.cpuct * Ps_root[a] * math.sqrt(S) / gap - 1)
                    adjusted_counts[a] = min(n, max(n - n_forced, int(n_min), 0))


                adjusted_counts = [c if c > 1 else 0 for c in adjusted_counts]
                counts = adjusted_counts
            valid_moves_mask = self.nodes_data[s][1] # Vs from root node

        if sum(counts) <= 0:
            # defensive only: spread over legal moves rather than divide by zero
            counts = [int(x) for x in valid_moves_mask]

        # Clean search tree from very old moves = less memory footprint and less keys to search into
        if not self.args.no_mem_optim:
            r = self.game.getRound(canonicalBoard)
            if r > self.last_cleaning + 20:
                for node in [n for n in self.nodes_data.keys() if self.nodes_data[n][6] < r-5]:
                    del self.nodes_data[node]
                self.last_cleaning = int(r)

        if temp <= 0.02: # below this threshold the power below overflows
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs, q, is_full_search

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs, q, is_full_search

    def search(self, canonicalBoard, dirichlet_noise=False, forced_playouts=False, is_root=False):
        """
        One simulation: descends by highest UCB until a leaf, expands the leaf with
        the network (or reads the outcome of a terminal node), and backs the value
        up the path.

        Returns:
            v: per-player value vector, in the canonical order of canonicalBoard
        """
        s = self.game.stringRepresentation(canonicalBoard)
        Es, Vs, Ps, meta_ns_qs, Qsa, Nsa, r = self.nodes_data.get(s, (None, )*7)
        if r is None:
            r = self.game.getRound(canonicalBoard)

        if Es is None:
            Es = self.game.getGameEnded(canonicalBoard, 0)
            if Es.any():
                # terminal node (we can leave meta_ns_qs as None since it won't be expanded)
                self.nodes_data[s] = (Es, Vs, Ps, None, Qsa, Nsa, r)
                return Es
        elif Es.any():
            # terminal node
            return Es

        if Ps is None:
            # First time that we explore state s
            Vs = self.game.getValidMoves(canonicalBoard, 0)
            nn_board = self.game.getObservation(canonicalBoard, 0) if self.hidden_info else canonicalBoard
            if self.batch_info is None:
                Ps, v = self.nnet.predict(nn_board, Vs)
            else:
                Ps, v = self.nnet.predict_client(nn_board, Vs, self.batch_info)
            if dirichlet_noise:
                Ps = softmax(Ps, self.args.temperature[2])
                self.applyDirNoise(Ps, Vs)
            normalise(Ps)

            Qsa, Nsa = self.Qsa_default.copy(), self.Nsa_default.copy()

            # Mutable [Ns, Q_vector]: keep the full per-player Q (no zero-sum assumption)
            meta_ns_qs = [0, np.array(v, dtype=np.float64)]
            self.nodes_data[s] = (Es, Vs, Ps, meta_ns_qs, Qsa, Nsa, r)
            return v

        if dirichlet_noise:
            Ps = softmax(Ps, self.args.temperature[2])
            if Ps is self.nodes_data[s][2]:   # softmax_temp == 1.0 returns the same object
                Ps = Ps.copy()
            self.applyDirNoise(Ps, Vs)
            normalise(Ps)
            # Persist noised priors so sims 1..N-1 see them (a node is root at most
            # once: the state vector embeds the move counter, so this is never revisited)
            self.nodes_data[s] = (Es, Vs, Ps, meta_ns_qs, Qsa, Nsa, r)

        Ns, Qs = meta_ns_qs[0], float(meta_ns_qs[1][0])   # scalar Q for numba FPU

        # pick the action with the highest upper confidence bound
        # get next state and get canonical version of it
        a, next_s, next_player = get_next_best_action_and_canonical_state(
            Es, Vs, Ps, Ns, Qsa, Nsa, Qs,
            self.args.cpuct,
            self.game.board,
            canonicalBoard,
            forced_playouts,
            is_root,
            self.step,
            self.args.fpu,
            self.args.fpu_root,
            self.random_seed,
            self.args.forced_playouts_k,
        )

        v = self.search(next_s)
        v = np_roll(v, next_player)

        Qsa[a] = (Nsa[a] * Qsa[a] + v[0]) / (Nsa[a] + 1) # if Qsa[a] is NAN, then Nsa is zero
        meta_ns_qs[1] = ((meta_ns_qs[0]+1) * meta_ns_qs[1] + v) / (meta_ns_qs[0]+2)
        Nsa[a] += 1
        meta_ns_qs[0] += 1
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
        gc.collect()

@njit(cache=True, fastmath=True, nogil=True)
def np_roll(arr, n):
    return np.roll(arr, n)

# pick the action with the highest upper confidence bound
@njit(cache=True, fastmath=True, nogil=True)
def pick_highest_UCB(Es, Vs, Ps, Ns, Qsa, Nsa, Qs, cpuct, forced_playouts, is_root, n_iter, fpu, fpu_root, k):
    cur_best = MINFLOAT
    best_act = -1

    # Apply a specific FPU reduction (usually none) if we are at the root node
    fpu_init = Qs - fpu_root if is_root else Qs - fpu

    for a, valid in enumerate(Vs):
        if valid:
            if forced_playouts:
                if Nsa[a] < int(math.sqrt(k * Ps[a] * n_iter)):
                    u = 1000000.0 + Ps[a]
                    if u > cur_best:
                        cur_best, best_act = u, a
                    continue

            if Qsa[a] != NAN:
                u = Qsa[a] + cpuct * Ps[a] * math.sqrt(Ns) / (1 + Nsa[a])
            else:
                u = fpu_init + cpuct * Ps[a] * math.sqrt(Ns + EPS)

            if u > cur_best:
                cur_best, best_act = u, a

    return best_act


@njit(fastmath=True, nogil=True) # no cache because it relies on jitclass which isn't compatible with cache
def get_next_best_action_and_canonical_state(Es, Vs, Ps, Ns, Qsa, Nsa, Qs, cpuct, gameboard, canonicalBoard, forced_playouts, is_root, n_iter, fpu, fpu_root, random_seed, k):
    a = pick_highest_UCB(Es, Vs, Ps, Ns, Qsa, Nsa, Qs, cpuct, forced_playouts, is_root, n_iter, fpu, fpu_root, k)

    # Do action 'a'
    gameboard.copy_state(canonicalBoard, True)
    next_player = gameboard.make_move(a, 0, random_seed=random_seed)

    # Get canonical form
    if next_player != 0:
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
    result = (Ps + 1e-12) ** (1. / softmax_temp)
    normalise(result)
    return result.astype(np.float32)
