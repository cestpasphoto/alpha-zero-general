"""
CatanTrade.py -- player-trade helpers shared by the search and by the diagnostics.

  - board reading and offer enumeration (every legal ask x give),
  - Evaluator: batched net values of (board, player to move) pairs,
  - score_offers(): one-ply EV of every offer over worlds dealt from the
    proposer's observation (used by trade_scan.py),
  - A: is_answer_node(), the root prior of a trade answer is flattened (MCTS
    arg flat_answer),
  - B: root_move_filter(), at a trade decision the root keeps a single offer,
    chosen against responders that SEARCH, or none (MCTS arg trade_filter).

Play only: both are opt-in MCTS args, which Coach never sets, so self-play and
the arena gate are unchanged.
"""
import atexit
import math
import time

import numpy as np

try:
	from .CatanConstants import (
		N_PLAYERS, N_RESOURCES, N_TRADE_SETS, TRADE_SETS,
		ROW_PLAYER, ROW_GLOBAL, GA_PHASE, PA_RESOURCES,
		PD_TRADE_RECV, PD_TRADE_STATUS, TRADE_COMPOSING,
		PHASE_MAIN, PHASE_TRADE_OFFER, PHASE_TRADE_ANSWER,
		A_TRADE_RECV, A_TRADE_GIVE, A_TRADE_OK, A_TRADE_NO,
	)
except ImportError:
	from CatanConstants import (
		N_PLAYERS, N_RESOURCES, N_TRADE_SETS, TRADE_SETS,
		ROW_PLAYER, ROW_GLOBAL, GA_PHASE, PA_RESOURCES,
		PD_TRADE_RECV, PD_TRADE_STATUS, TRADE_COMPOSING,
		PHASE_MAIN, PHASE_TRADE_OFFER, PHASE_TRADE_ANSWER,
		A_TRADE_RECV, A_TRADE_GIVE, A_TRADE_OK, A_TRADE_NO,
	)


RES = 'BLOGW'


class dotdict(dict):
	def __getattr__(self, name):
		return self[name]


def _mcts():
	# imported lazily: MCTS.py lives at the repository root and knows nothing of Catan
	import MCTS as mcts_mod
	return mcts_mod


# =============================================================================
# Board reading (canonical frame, player 0 = the one to move)
# =============================================================================

def phase(b):
	return int(b[ROW_GLOBAL, GA_PHASE])


def hand(b, p):
	return b[ROW_PLAYER + 4 * p, PA_RESOURCES:PA_RESOURCES + N_RESOURCES].astype(np.int64)


def composing_ask(b):
	"""Ask of player 0 if it is composing an offer (GIVE ply pending), else None."""
	d = ROW_PLAYER + 3
	if phase(b) != PHASE_TRADE_OFFER or b[d, PD_TRADE_STATUS] != TRADE_COMPOSING:
		return None
	recv = b[d, PD_TRADE_RECV:PD_TRADE_RECV + N_RESOURCES]
	if recv.sum() == 0:
		return None
	hits = np.flatnonzero((TRADE_SETS == recv).all(axis=1))
	return int(hits[0]) if len(hits) else None


def set_label(s):
	return '+'.join(f'{int(n)}{RES[r]}' for r, n in enumerate(TRADE_SETS[s]) if n > 0)


def offer_label(a, g):
	return f'ask {set_label(a)} / give {set_label(g)}'


# =============================================================================
# Batched value evaluation
# =============================================================================

class Evaluator:
	"""
	Net values of many (board, player to move) pairs, in the frame of the
	boards. Batched through the ONNX session when the net exposes one (the
	GenericNNetWrapper does), with a parity check against net.predict on the
	first batch; one net.predict per board otherwise.
	"""

	def __init__(self, game, net, chunk=1024):
		self.game, self.net, self.chunk = game, net, chunk
		self.cache = {}
		self.checked = False
		self.n_evals = 0

	def _session(self):
		if hasattr(self.net, 'switch_target'):
			self.net.switch_target('inference')
		if getattr(self.net, 'current_mode', None) == 'onnx':
			return getattr(self.net, 'ort_session', None)
		return None

	def _predict_many(self, obs, valids):
		sess = self._session()
		if sess is None:
			return np.array([self.net.predict(o, v)[1] for o, v in zip(obs, valids)], dtype=np.float64)
		out = []
		for i in range(0, len(obs), self.chunk):
			res = sess.run(None, {
				'board': np.stack(obs[i:i + self.chunk]).astype(np.float32),
				'valid_actions': np.stack(valids[i:i + self.chunk]).astype(np.bool_),
			})
			out.append(np.asarray(res[1], dtype=np.float64))
		v = np.concatenate(out)
		if not self.checked:   # the batched path must compute what net.predict computes
			for k in range(min(8, len(obs))):
				ref = np.asarray(self.net.predict(obs[k], valids[k])[1], dtype=np.float64)
				if np.abs(ref - v[k]).max() > 1e-4:
					raise RuntimeError(f'batched ONNX values differ from net.predict: {ref} vs {v[k]}')
			self.checked = True
		return v

	def values(self, items):
		"""items: list of (board, mover) in some frame F. Returns value vectors in frame F."""
		g = self.game
		res = [None] * len(items)
		todo, obs, valids, movers, keys = [], [], [], [], []
		first = {}                       # same state twice in one batch: evaluate once
		dup = []
		for i, (b, m) in enumerate(items):
			key = (b.tobytes(), m)
			if key in self.cache:
				res[i] = self.cache[key]
				continue
			if key in first:
				dup.append((i, key))
				continue
			first[key] = i
			ended = g.getGameEnded(b, m)
			if np.any(ended):
				res[i] = self.cache[key] = np.asarray(ended, dtype=np.float64)
				continue
			cb = np.copy(g.getCanonicalForm(b, m))
			todo.append(i)
			obs.append(np.copy(g.getObservation(cb, 0)))
			valids.append(np.array(g.getValidMoves(cb, 0), dtype=np.bool_))
			movers.append(m)
			keys.append(key)
		if todo:
			v = self._predict_many(obs, valids)
			self.n_evals += len(todo)
			for j, i in enumerate(todo):
				val = np.roll(v[j], movers[j])      # canonical of the mover -> frame F
				self.cache[keys[j]] = val
				res[i] = val
		for i, key in dup:
			res[i] = self.cache[key]
		return res


# =============================================================================
# Offers
# =============================================================================

def legal_gives(b_proposer, ask):
	"""Mirror of Board._give_is_legal (own hand, disjoint from the ask)."""
	my = hand(b_proposer, 0)
	out = []
	for g in range(N_TRADE_SETS):
		if (TRADE_SETS[g] > my).any():
			continue
		if ((TRADE_SETS[g] > 0) & (TRADE_SETS[ask] > 0)).any():
			continue
		out.append(g)
	return out


def enumerate_offers(game, cb):
	"""Every legal (ask, give) for the player to move, and whether the ask is already made."""
	fixed = composing_ask(cb)
	if fixed is not None:
		asks, at_give = [fixed], True
	elif phase(cb) == PHASE_MAIN:
		valid = game.getValidMoves(cb, 0)
		asks, at_give = [s for s in range(N_TRADE_SETS) if valid[A_TRADE_RECV + s]], False
	else:
		return None, None
	offers = [(a, g) for a in asks for g in legal_gives(cb, a)]
	return offers, at_give


def after_offer(game, w, a, g, at_give, n_gives, seed):
	"""Play ask + give on world w (proposer frame). Returns (board, mover)."""
	b, m = np.copy(w), 0
	if not at_give:
		b, m = game.getNextState(b, 0, A_TRADE_RECV + a, random_seed=seed)
		b, m = np.copy(b), int(m)
	if m == 0 and composing_ask(b) is not None:
		valid = game.getValidMoves(game.getCanonicalForm(b, 0), 0)
		assert valid[A_TRADE_GIVE + g], f'give {set_label(g)} illegal after ask {set_label(a)}'
		b, m = game.getNextState(b, 0, A_TRADE_GIVE + g, random_seed=seed)
		b, m = np.copy(b), int(m)
	else:
		# make_move resolved the GIVE ply on its own: it had a single legal option
		assert n_gives == 1, 'GIVE ply skipped although several gives are legal'
	return b, m


def check_gives(game, w, offers, at_give, seed):
	"""Assert once per position that legal_gives() matches valid_moves()."""
	for a in sorted({a for a, _ in offers})[:5]:
		b, m = np.copy(w), 0
		if not at_give:
			b, m = game.getNextState(b, 0, A_TRADE_RECV + a, random_seed=seed)
			b, m = np.copy(b), int(m)
		if m != 0 or composing_ask(b) is None:
			continue
		valid = game.getValidMoves(game.getCanonicalForm(b, 0), 0)
		mine = set(legal_gives(b, a))
		theirs = {g for g in range(N_TRADE_SETS) if valid[A_TRADE_GIVE + g]}
		if mine != theirs:
			raise RuntimeError(f'legal_gives mismatch after ask {set_label(a)}: {sorted(mine ^ theirs)}')


def resolve(game, ev, starts, margin, seed):
	"""
	Walk the answers of every (board, mover) right after an offer, all of them in
	lockstep so that each round of answers is one batch of evaluations.
	A responder accepts iff v_self(after OK) - v_self(after NO) > margin.
	Returns per start: (accepter relative to the proposer or 0, final board, final mover,
	gain of the accepter).
	"""
	P = game.num_players
	cur = list(starts)
	out = [None] * len(starts)
	pending = []
	for i, (b, m) in enumerate(cur):
		if phase(b) == PHASE_TRADE_ANSWER and m != 0:
			pending.append(i)
		else:
			out[i] = (0, b, m, 0.)
	while pending:
		ok_states, no_states = [], []
		for i in pending:
			b, m = cur[i]
			cm = game.getCanonicalForm(b, m)
			valid = game.getValidMoves(cm, 0)
			assert valid[A_TRADE_OK] and valid[A_TRADE_NO], 'exposed answer node without both OK and NO'
			bo, mo = game.getNextState(b, m, A_TRADE_OK, random_seed=seed)
			ok_states.append((np.copy(bo), int(mo)))
			bn, mn = game.getNextState(b, m, A_TRADE_NO, random_seed=seed)
			no_states.append((np.copy(bn), int(mn)))
		v_ok = ev.values(ok_states)
		v_no = ev.values(no_states)
		nxt = []
		for j, i in enumerate(pending):
			p = cur[i][1]
			d = v_ok[j][p] - v_no[j][p]
			if d > margin:
				out[i] = (p, ok_states[j][0], ok_states[j][1], float(d))
				continue
			bn, mn = no_states[j]
			cur[i] = (bn, mn)
			if phase(bn) == PHASE_TRADE_ANSWER and mn != 0:
				nxt.append(i)
			else:
				out[i] = (0, bn, mn, 0.)
		pending = nxt
	return out


def score_offers(game, ev, cb, offers, at_give, seeds, margin):
	"""
	EV and acceptance of each offer over the worlds dealt with `seeds` from the
	proposer's observation. Reusable as is for a root-level offer filter (L2).
	"""
	obs = np.copy(game.getObservation(cb, 0))
	worlds = [np.copy(game.sampleWorld(obs, int(s))) for s in seeds]
	n_gives = {}
	for a, _ in offers:
		n_gives[a] = n_gives.get(a, 0) + 1
	check_gives(game, worlds[0], offers, at_give, int(seeds[0]))

	K, Wn = len(offers), len(worlds)
	gain = np.zeros((K, Wn))
	acc = np.zeros((K, Wn), dtype=np.int64)       # accepter, relative to the proposer; 0 = nobody
	acc_gain = np.full((K, Wn), np.nan)           # what the accepter gains, by its own net value
	base_v = np.zeros(Wn)
	for w, (world, s) in enumerate(zip(worlds, seeds)):
		s = int(s)
		starts = [after_offer(game, world, a, g, at_give, n_gives[a], s) for a, g in offers]
		res = resolve(game, ev, starts, margin, s)
		# baseline: every responder declines. It is the same state whatever the
		# offer (the offer rows are cleared, the attempt is spent), so take it
		# from any offer by declining all the way, and check it is unique.
		declined = resolve(game, ev, starts[:1], float('inf'), s)[0]
		base = ev.values([(declined[1], declined[2])])[0]
		base_key = declined[1].tobytes()
		base_v[w] = base[0]
		finals = ev.values([(r[1], r[2]) for r in res])
		for k, r in enumerate(res):
			if r[0] == 0 and r[1].tobytes() != base_key:
				raise RuntimeError(f'declined offer {offer_label(*offers[k])} does not reach the common baseline state')
			acc[k, w] = r[0]
			gain[k, w] = finals[k][0] - base[0]
			if r[0] != 0:
				acc_gain[k, w] = r[3]
	return dict(ev=gain.mean(axis=1), p_acc=(acc > 0).mean(axis=1), gain=gain, acc=acc,
	            acc_gain=acc_gain, base_v=base_v, worlds=worlds)


# =============================================================================
# Verification with the real search
# =============================================================================

def answer_search(game, net, b, m, args):
	"""Real MCTS answer of responder m on board b (proposer frame): visit shares
	and the root statistics of its own (single, u=1) search tree."""
	mcts_mod = _mcts()
	cm = np.copy(game.getCanonicalForm(b, m))
	mcts = mcts_mod.MCTS(game, net, args)
	probs, _, _ = mcts.getActionProb(cm, temp=1, force_full_search=True)
	key = game.stringRepresentation(game.sampleWorld(game.getObservation(cm, 0), mcts_mod.magic_seeds[0]))
	node = mcts.nodes_data.get(key)
	if node is None:        # fall back on the most visited node, which is the root
		live = [n for n in mcts.nodes_data.values() if n[3] is not None]
		node = max(live, key=lambda n: n[3][0]) if live else None
	if node is None or node[3] is None:
		# The responder's own world is TERMINAL: the turn player (the proposer)
		# reaches 10 VP with invented VP cards. MCTS then falls back on uniform
		# counts, i.e. the AI answers at random (temp=0 tie-break). Counted apart.
		nan = float('nan')
		return dict(responder=int(m), share_ok=float(probs[A_TRADE_OK]), n_ok=0, n_no=0,
		            q_ok=nan, q_no=nan, prior_ok=nan, terminal_root=True)
	Ps, Qsa, Nsa = node[2], node[4], node[5]
	q = lambda a: float(Qsa[a]) if Nsa[a] > 0 else float('nan')
	return dict(responder=int(m), share_ok=float(probs[A_TRADE_OK]),
	            n_ok=int(Nsa[A_TRADE_OK]), n_no=int(Nsa[A_TRADE_NO]),
	            q_ok=q(A_TRADE_OK), q_no=q(A_TRADE_NO), prior_ok=float(Ps[A_TRADE_OK]), terminal_root=False)


# =============================================================================
# A -- flat prior at the root of a trade answer
# =============================================================================
# Self-play has taught a strong NO prior (P(OK) ~0.2-0.3). With 2 children,
# OK then needs Q(OK) - Q(NO) > ~cpuct*sqrt(N)*(1-2P)/(1+N/2) (0.06 at m=200)
# to be the most visited: the answer follows the prior, not the responder's
# own Q. Flattening the ROOT prior makes the visits follow Q. Inner answer
# nodes (how the proposer's tree models the responders) are left alone.

def is_answer_node(board):
	return phase(board) == PHASE_TRADE_ANSWER


# =============================================================================
# B -- root offer filter
# =============================================================================
# At a trade decision of the player to move:
#   ask ply  : the one-ply rule pre-selects the `trade_k` best offers over
#              `trade_sel_worlds` worlds; each is then played in `trade_worlds`
#              worlds of the proposer's belief against responders that answer
#              with a real search (`trade_resp_sims`, eval profile, deciding on
#              visits like the baseline AI). The best one is kept if its mean
#              gain over "everybody declines" exceeds `trade_min_gain`; the root
#              then allows that single ask and no other. Otherwise no ask at all.
#   give ply : the give of the plan chosen at the ask ply, recomputed if the
#              plan does not match the standing ask.
# The rest of the root (build, buy, end turn...) is untouched: the search still
# decides WHETHER to trade now.

STATS = dict(ask_calls=0, allowed=0, no_candidate=0, gains=[], give_calls=0, give_recomputed=0, seconds=0.)


@atexit.register
def _report():
	if STATS['ask_calls'] == 0:
		return
	g = np.array(STATS['gains']) if STATS['gains'] else np.zeros(1)
	n = STATS['ask_calls']
	print(f'[trade filter] {n} ask decisions filtered: offer allowed {STATS["allowed"] / n:.0%}, no candidate '
	      f'after the one-ply rule {STATS["no_candidate"] / n:.0%}; searched gain of the best candidate: median '
	      f'{np.median(g):+.4f}, mean {g.mean():+.4f}; {STATS["give_calls"]} give plies '
	      f'({STATS["give_recomputed"]} recomputed); {STATS["seconds"]:.0f} s spent, '
	      f'{STATS["seconds"] / n:.2f} s per ask decision')


def _opt(args, key, default):
	try:
		v = args[key] if isinstance(args, dict) else getattr(args, key, default)
	except KeyError:
		return default
	return default if v is None else v


def _responder_args(args, sims):
	return dotdict(numMCTSSims=int(sims), cpuct=_opt(args, 'cpuct', 1.0), fpu=_opt(args, 'fpu', 0.1),
	               fpu_root=0.0, universes=1, prob_fullMCTS=1., ratio_fullMCTS=1, forced_playouts=False,
	               forced_playouts_k=1.5, no_mem_optim=True, dirichletAlpha=0., temperature=[1., 1., 1., 1.])


def searched_gain(game, net, ev, world, a, g, at_give, n_gives, seed, resp_args):
	"""Proposer's gain over "everybody declines" when the offer meets responders
	that answer with a real search, deciding on visits. world: proposer frame."""
	b, m = after_offer(game, world, a, g, at_give, n_gives, seed)
	declined = resolve(game, ev, [(b, m)], float('inf'), seed)[0]
	base = ev.values([(declined[1], declined[2])])[0][0]
	mcts_mod = _mcts()
	while phase(b) == PHASE_TRADE_ANSWER and m != 0:
		cm = np.copy(game.getCanonicalForm(b, m))
		probs = mcts_mod.MCTS(game, net, resp_args).getActionProb(cm, temp=1, force_full_search=True)[0]
		ok = probs[A_TRADE_OK] > probs[A_TRADE_NO]
		b, m = game.getNextState(b, m, A_TRADE_OK if ok else A_TRADE_NO, random_seed=seed)
		b, m = np.copy(b), int(m)
		if ok:
			break
	return float(ev.values([(b, m)])[0][0] - base)


def choose_offer(game, net, cb, args):
	"""Best offer (ask, give, searched gain) for the player to move on cb, or None."""
	offers, at_give = enumerate_offers(game, cb)
	if not offers:
		return None, at_give
	rng = np.random.default_rng()
	ev = Evaluator(game, net)
	n_gives = {}
	for a, _ in offers:
		n_gives[a] = n_gives.get(a, 0) + 1
	S = score_offers(game, ev, cb, offers, at_give,
	                 rng.integers(100, 2**31 - 2, size=int(_opt(args, 'trade_sel_worlds', 16))), 0.0)
	# Only offers that some world accepts under the one-ply rule are worth a
	# search: the rule accepts MORE than the search (58% vs 18-24% measured), so
	# an offer it rejects everywhere is very unlikely to pass real responders.
	# This skips the expensive part in most positions.
	ranked = [k for k in np.argsort(-S['ev']) if S['p_acc'][k] > 0 and S['ev'][k] > 0]
	top = ranked[:int(_opt(args, 'trade_k', 3))]
	if not top:
		return None, at_give
	obs = np.copy(game.getObservation(cb, 0))
	wseeds = rng.integers(100, 2**31 - 2, size=int(_opt(args, 'trade_worlds', 4)))
	worlds = [np.copy(game.sampleWorld(obs, int(s))) for s in wseeds]
	resp_args = _responder_args(args, _opt(args, 'trade_resp_sims', _opt(args, 'numMCTSSims', 200)))
	best = None
	for k in top:
		a, g = offers[k]
		gain = float(np.mean([searched_gain(game, net, ev, w, a, g, at_give, n_gives[a], int(s), resp_args)
		                      for w, s in zip(worlds, wseeds)]))
		if best is None or gain > best[2]:
			best = (int(a), int(g), gain)
	return best, at_give


def root_move_filter(game, net, cb, args, memo):
	"""Mask of the moves the root may play (True = allowed), or None to leave it alone."""
	ph = phase(cb)
	valid = np.array(game.getValidMoves(cb, 0), dtype=bool)
	if ph == PHASE_MAIN and valid[A_TRADE_RECV:A_TRADE_RECV + N_TRADE_SETS].any():
		t0 = time.time()
		best, _ = choose_offer(game, net, cb, args)
		STATS['ask_calls'] += 1
		STATS['seconds'] += time.time() - t0
		mask = valid.copy()
		mask[A_TRADE_RECV:A_TRADE_RECV + N_TRADE_SETS] = False
		memo.pop('plan', None)
		if best is None:
			STATS['no_candidate'] += 1
		else:
			STATS['gains'].append(best[2])
			if best[2] > float(_opt(args, 'trade_min_gain', 0.01)):
				mask[A_TRADE_RECV + best[0]] = True
				memo['plan'] = best
				STATS['allowed'] += 1
		return mask
	ask = composing_ask(cb)
	if ask is not None:
		STATS['give_calls'] += 1
		plan = memo.pop('plan', None)
		if plan is None or plan[0] != ask or not valid[A_TRADE_GIVE + plan[1]]:
			t0 = time.time()
			plan, _ = choose_offer(game, net, cb, args)
			STATS['give_recomputed'] += 1
			STATS['seconds'] += time.time() - t0
		mask = valid.copy()
		mask[A_TRADE_GIVE:A_TRADE_GIVE + N_TRADE_SETS] = False
		mask[A_TRADE_GIVE + plan[1]] = True
		return mask
	return None

