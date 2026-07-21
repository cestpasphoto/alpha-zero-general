import logging
import math
import numpy as np
import gc
from random import randrange
from copy import deepcopy

from numba import njit

EPS = 1e-8
NAN = -42.
MINFLOAT = float('-inf')
magic_seeds = [31416, 1, 14142, 42, 27183, 2, 16180, 7]

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
        self.Qsa_default = np.full (self.game.getActionSize(), NAN, dtype=np.float32)
        self.Nsa_default = np.zeros(self.game.getActionSize()     , dtype=np.int16)

        self.rng = np.random.default_rng()
        self.step = 0
        self.last_cleaning = 0
        self.batch_info = batch_info
        self.random_seed = -1
        self.max_current_depth = 0
        self.sum_new_nodes_depth = 0

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
        # KataGo regime: self-play full searches start from a fresh tree so FP/PTP/noise
        # operate on uncontaminated counts. Tree reuse stays on elsewhere (fast searches,
        # arena, pit) because self.dirichlet_noise is False there.
        if is_full_search and self.dirichlet_noise:
            self.nodes_data = {}
            self.last_cleaning = 0
        initial_nodes_count = len(self.nodes_data)
        self.max_current_depth = 0
        self.sum_new_nodes_depth = 0

        # Gumbel root Sequential Halving (training-only, replaces the whole root
        # machinery below: Dirichlet noise, forced playouts, PTP and visit-count
        # targets). Gated on self.dirichlet_noise so that pit/arena/pnet MCTS
        # instances (dirichlet_noise=False) are never affected, and on
        # is_full_search so that PCR fast searches keep the cheap PUCT path.
        # getattr: robust to eval-side dotdict args that lack the 'gumbel' key.
        if getattr(self.args, 'gumbel', False) and is_full_search and self.dirichlet_noise:
            return self._gumbel_root_search(canonicalBoard, nb_MCTS_sims, initial_nodes_count)

        for self.step in range(nb_MCTS_sims):
            self.random_seed = magic_seeds[self.step % self.args.universes] if self.args.universes > 0 else -1
            dir_noise = (self.step == 0 and is_full_search and self.dirichlet_noise)
            self.search(canonicalBoard, dirichlet_noise=dir_noise, forced_playouts=forced_playouts, is_root=True, depth=0)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.nodes_data[s][5][a] for a in range(self.game.getActionSize())] # Nsa

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
                new_n = max(n - n_forced, int(n_min), 0)
                adjusted_counts[a] = new_n
            adjusted_counts = [c if c > 1 else 0 for c in adjusted_counts]
            counts = adjusted_counts

        probs = np.array(counts)
        probs = probs / probs.sum()

        # Metrics
        new_nodes = len(self.nodes_data) - initial_nodes_count
        entropy = -np.sum(probs * np.log(probs + 1e-8)) # 1e-8 to avoid log(0)
        confidence = float(np.max(probs))
        avg_new_depth = (self.sum_new_nodes_depth / new_nodes) if new_nodes > 0 else 0.0
        valid_moves_mask = self.nodes_data[s][1] # Vs from root node
        total_valid_moves = np.sum(valid_moves_mask)
        visited_at_root = sum(1 for a in range(self.game.getActionSize()) if valid_moves_mask[a] and counts[a] > 0)
        root_coverage = (visited_at_root / total_valid_moves) if total_valid_moves > 0 else 0.0

        metrics = {
            "max_depth": self.max_current_depth,
            "avg_new_depth": avg_new_depth,
            "new_nodes": new_nodes,
            "entropy": entropy,
            "confidence": confidence,
            "root_coverage": root_coverage,
        }

        # Clean search tree from very old moves = less memory footprint and less keys to search into
        if not self.args.no_mem_optim:
            r = self.game.getRound(canonicalBoard)
            if r > self.last_cleaning + 20:
                for node in [n for n in self.nodes_data.keys() if self.nodes_data[n][6] < r-5]:
                    del self.nodes_data[node]
                self.last_cleaning = int(r)

        if temp <= 0.02: # For temp below this threshold it gives an overflow error in the next power operatio
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs, q, is_full_search, metrics

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs, q, is_full_search, metrics

    def _gumbel_root_search(self, canonicalBoard, nb_MCTS_sims, initial_nodes_count):
        """
        Gumbel AlphaZero root procedure (Danihelka et al. 2022, "Policy improvement
        by planning with Gumbel"): Sequential Halving with Gumbel over the top-m
        root actions, then a completed-Q improved policy as training target.

        Root-only hybrid: non-root selection stays standard PUCT (the usual
        pragmatic setup, e.g. in mctx-based reimplementations). At the root this
        REPLACES Dirichlet noise, forced playouts, PTP and visit-count targets.

        Returns the same 4-tuple as getActionProb, plus metrics['gumbel_action']:
        the Sequential Halving winner, which the Coach must PLAY as-is.
        Exploration comes from the Gumbel noise (resampled at every move), not
        from temperature sampling of the returned policy.

        Approximation vs the paper: unvisited actions are completed with the root
        running-mean value instead of the exact v_mix interpolation. The running
        mean already blends the raw net value (its initialisation) with search
        returns, which is the same intent.
        """
        action_size = self.game.getActionSize()
        c_visit = float(self.args.gumbel_cvisit)
        c_scale = float(self.args.gumbel_cscale)

        # --- Expand the root if needed; this consumes one simulation (honest budget) ---
        s = self.game.stringRepresentation(canonicalBoard)
        sims_done = 0
        if self.nodes_data.get(s, (None,)*7)[2] is None:
            self.step = 0
            self.random_seed = magic_seeds[0] if self.args.universes > 0 else -1
            self.search(canonicalBoard, is_root=True, depth=0)
            sims_done = 1
        Es, Vs, Ps, meta_ns_qs, Qsa, Nsa, r = self.nodes_data[s]
        if Ps is None:
            raise ValueError('Gumbel root search called on a terminal state')

        valid_idx = np.flatnonzero(np.asarray(Vs))
        logits = np.log(np.asarray(Ps, dtype=np.float64) + 1e-12)

        # --- Gumbel noise: sampled ONCE per move, shared by candidate selection,
        #     halving comparisons and the final argmax (required by the theory) ---
        g = self.rng.gumbel(size=action_size)
        root_scores = g + logits

        # --- Candidate set: top-m legal actions by g + logits ---
        m = int(min(max(1, self.args.gumbel_m), len(valid_idx)))
        cand = valid_idx[np.argsort(root_scores[valid_idx])[::-1][:m]].tolist()

        def q_hat(a):
            # Backed-up Q if visited (already from the current player's viewpoint,
            # cf. the np_roll in search), root running-mean value otherwise.
            return float(Qsa[a]) if Nsa[a] > 0 else float(meta_ns_qs[1][0])

        def sh_score(a):
            # g(a) + logits(a) + sigma(q_hat(a)), sigma from section 4 of the paper
            max_visit = int(Nsa[valid_idx].max())
            return root_scores[a] + (c_visit + max_visit) * c_scale * q_hat(a)

        def one_forced_sim(a):
            self.step = sims_done  # only feeds the (disabled) FP quota, kept coherent anyway
            self.random_seed = magic_seeds[sims_done % self.args.universes] if self.args.universes > 0 else -1
            self.search(canonicalBoard, is_root=True, depth=0, force_action=int(a))

        # --- Sequential Halving over the remaining budget ---
        n_phases = max(1, int(math.ceil(math.log2(m)))) if m > 1 else 1
        for phase in range(n_phases):
            if sims_done >= nb_MCTS_sims or not cand:
                break
            m_k = len(cand)
            phases_left = n_phases - phase
            per_action = max(1, (nb_MCTS_sims - sims_done) // max(1, phases_left * m_k))
            for a in cand:
                for _ in range(per_action):
                    if sims_done >= nb_MCTS_sims:
                        break
                    one_forced_sim(a)
                    sims_done += 1
            if phase < n_phases - 1 and len(cand) > 1:
                cand = sorted(cand, key=sh_score, reverse=True)[:max(1, (len(cand) + 1) // 2)]

        # Leftover budget (integer-division remainders): round-robin on the finalists
        i = 0
        while sims_done < nb_MCTS_sims and cand:
            one_forced_sim(cand[i % len(cand)])
            sims_done += 1
            i += 1

        chosen_a = int(max(cand, key=sh_score)) if cand else int(valid_idx[np.argmax(root_scores[valid_idx])])

        # --- Improved policy target: softmax(logits + sigma(completedQ)) on legal actions ---
        max_visit = int(Nsa[valid_idx].max())
        v_root = float(meta_ns_qs[1][0])
        completed_q = np.where(np.asarray(Nsa) > 0, np.asarray(Qsa, dtype=np.float64), v_root)
        pi_logits = (logits + (c_visit + max_visit) * c_scale * completed_q)[valid_idx]
        pi_valid = np.exp(pi_logits - pi_logits.max())
        pi_valid /= pi_valid.sum()
        probs = np.zeros(action_size, dtype=np.float64)
        probs[valid_idx] = pi_valid

        # Per-player Q measured directly from backups, same as the standard path
        q = list(meta_ns_qs[1])

        # Metrics: same keys as the standard path (entropy/confidence computed on
        # the improved policy; root_coverage is bounded by gumbel_m / nb valids by
        # design, do not compare it against non-Gumbel runs), plus the SH winner.
        new_nodes = len(self.nodes_data) - initial_nodes_count
        total_valid = int(len(valid_idx))
        visited_at_root = int(np.sum(np.asarray(Nsa)[valid_idx] > 0))
        metrics = {
            "max_depth": self.max_current_depth,
            "avg_new_depth": (self.sum_new_nodes_depth / new_nodes) if new_nodes > 0 else 0.0,
            "new_nodes": new_nodes,
            "entropy": float(-np.sum(pi_valid * np.log(pi_valid + 1e-8))),
            "confidence": float(pi_valid.max()),
            "root_coverage": (visited_at_root / total_valid) if total_valid > 0 else 0.0,
            "gumbel_action": chosen_a,
        }
        return list(probs), q, True, metrics

    def search(self, canonicalBoard, dirichlet_noise=False, forced_playouts=False, is_root=False, depth=0, force_action=-1):
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
        if depth > self.max_current_depth:
            self.max_current_depth = depth

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
            self.sum_new_nodes_depth += depth
            Vs = self.game.getValidMoves(canonicalBoard, 0)
            if self.batch_info is None:
                Ps, v = self.nnet.predict(canonicalBoard, Vs)
            else:
                Ps, v = self.nnet.predict_client(canonicalBoard, Vs, self.batch_info)
            if dirichlet_noise:
                Ps = softmax(Ps, self.args.temperature[2])
                self.applyDirNoise(Ps, Vs)
            normalise(Ps)

            Qsa, Nsa = self.Qsa_default.copy(), self.Nsa_default.copy()
            
            # Mutable [Ns, Q_vector]: keep the full per-player Q (lives in Z-space,
            # no zero-sum assumption) instead of collapsing to player 0.
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
            force_action,
        )

        v = self.search(next_s, depth=depth+1)
        v = np_roll(v, next_player)

        Qsa[a] = (Nsa[a] * Qsa[a] + v[0]) / (Nsa[a] + 1) # if Qsa[a] is NAN, then Nsa is zero
        
        # In-place update of the full per-player Q vector
        # Qs = ((Ns+1) * Qs + v[0]) / (Ns+2)
        meta_ns_qs[1] = ((meta_ns_qs[0]+1) * meta_ns_qs[1] + v) / (meta_ns_qs[0]+2)
        Nsa[a] += 1
        # Ns += 1
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
def get_next_best_action_and_canonical_state(Es, Vs, Ps, Ns, Qsa, Nsa, Qs, cpuct, gameboard, canonicalBoard, forced_playouts, is_root, n_iter, fpu, fpu_root, random_seed, k, forced_action):
    # forced_action >= 0: Gumbel Sequential Halving dictates the root action,
    # bypassing PUCT entirely (root only; deeper calls always pass -1)
    if forced_action >= 0:
        a = forced_action
    else:
        a = pick_highest_UCB(Es, Vs, Ps, Ns, Qsa, Nsa, Qs, cpuct, forced_playouts, is_root, n_iter, fpu, fpu_root, k)

    # Do action 'a'
    gameboard.copy_state(canonicalBoard, True)
    next_player = gameboard.make_move(a, 0, random_seed=random_seed)
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
    result = (Ps + 1e-12) ** (1. / softmax_temp)
    normalise(result)
    return result.astype(np.float32)
