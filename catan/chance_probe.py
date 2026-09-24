"""Where does the noise in the search targets come from, and does the search
feed the network the kind of input it was trained on?

At -u 1 every search plans against ONE universe: the opponents' hidden hands
are invented once (sampleWorld) and every chance event (dice, development-card
draws, steals) is a fixed function of (seed, round, chance counter). MCTS.py
uses magic_seeds[0] = 31416 for that, in every search of every game. The
training target of a position is therefore the best move IN that universe,
which the network has no way to know. This probe replays the training search
on positions of a real buffer, changing ONE thing at a time, and measures how
much the target moves.

It also checks input coherence: Coach trains the network on getObservation()
(opponents' cards zeroed, so their `masked` flag is on), while MCTS.search()
evaluates it on the invented world (opponents' cards filled in, flag off).

Run from the REPO ROOT, like pit.py:
    python catan/chance_probe.py <checkpoint.examples> <checkpoint_N.pt> \
        [--positions 120] [--k 6] [--sims 800] [--iter -1] [--out probe.json]

Take the buffer and the checkpoint from the SAME run, the checkpoint being the
latest accepted one: the question is about the targets the loop produces now.

Sources of variation (K re-runs each, fresh tree, search profile read from the
checkpoint, no forced playouts -- they are off for hidden-info games anyway):
  universe   world AND chance stream re-drawn: what -u 1 bakes into a target
  chance     world fixed, chance stream re-drawn (dice, dev draws, steals)
  hands      chance fixed, invented opponent hands re-drawn
  dirichlet  universe fixed, root Dirichlet noise re-drawn (intended noise, for scale)
  sampled    world fixed, chance re-drawn at EVERY simulation: sampled chance
             nodes in one tree, i.e. a search that plans under uncertainty
  masked     like `universe`, but the network is queried on the mover's
             observation, as in training

Metrics (nats, the unit of the policy loss):
  dispersion = mean_k KL(pi_k || mean pi), x K/(K-1). The part of a KL policy
               loss that NO network can remove when the target is one draw of
               that source: the best prediction is the mean, its loss is this.
  bias       = JS(mean of one family, mean of another), next to its null (the
               same statistic between two halves of each family, halved).
"""
import argparse
import json
import os
import pickle
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, '.')
import MCTS as MCTS_mod                      # noqa: E402
from MCTS import MCTS                        # noqa: E402
from Stochastic import hashed_draw           # noqa: E402
from GameSwitcher import import_game         # noqa: E402

TRAINING_SEED = MCTS_mod.magic_seeds[0]      # the seed of every search at -u 1
PHASE_GROUPS = {0: 'setup', 1: 'setup', 2: 'roll', 3: 'discard/robber', 4: 'discard/robber',
                5: 'main', 6: 'main', 7: 'trade', 8: 'trade', 9: 'trade'}


# ---------------------------------------------------------------- loading
def _decode(e):
	return pickle.loads(zlib.decompress(e)) if isinstance(e, (bytes, bytearray)) else e


def load_iteration(path, index):
	"""checkpoint.examples = pickle of Coach.trainExamplesHistory: one container
	of examples per iteration, each a 5-tuple (board, pi, v, valids, q), raw or
	zlib-compressed."""
	with open(path, 'rb') as f:
		hist = list(pickle.load(f))
	print(f'{len(hist)} iterations in the buffer ({[len(i) for i in hist]}); using iteration {index}')
	return [_decode(e) for e in hist[index]]


def load_net(game, NNet, path):
	import torch
	ck = torch.load(path, map_location='cpu', weights_only=False)
	nn_version = ck['nn_version']
	net = NNet(game, dict(lr=None, dropout=0., epochs=None, batch_size=None, nn_version=nn_version))
	loaded = net.load_checkpoint(os.path.dirname(path) or '.', os.path.basename(path))
	if loaded is None:
		sys.exit(f'could not load {path}')
	return net, ck


def search_args(ck, sims):
	def get(k, default):
		v = ck.get(k)
		return default if v is None else v
	cpuct = get('cpuct', 1.25)
	cpuct = float(cpuct[0]) if isinstance(cpuct, (list, tuple)) else float(cpuct)
	return argparse.Namespace(
		numMCTSSims=int(sims or get('numMCTSSims', 800)), cpuct=cpuct,
		fpu=float(get('fpu', 0.1)), fpu_root=float(get('fpu_root', 0.0)),
		forced_playouts=False, forced_playouts_k=float(get('forced_playouts_k', 1.5)),
		dirichletAlpha=float(get('dirichletAlpha', -1)), temperature=list(get('temperature', [1.0, 0.1, 1.1, 10.0])),
		universes=1, prob_fullMCTS=1.0, ratio_fullMCTS=5, no_mem_optim=True)


# ---------------------------------------------------------------- statistics
def kl(p, q):
	m = p > 0
	return float(np.sum(p[m] * np.log(p[m] / np.maximum(q[m], 1e-12))))


def dispersion(ps, correct=True):
	"""K-way Jensen-Shannon = mean KL to the mean; x K/(K-1) for the small sample."""
	m = np.mean(ps, axis=0)
	d = float(np.mean([kl(p, m) for p in ps]))
	return d * len(ps) / (len(ps) - 1) if correct else d


def js(p, q):
	return dispersion([p, q], correct=False)


def top1_agreement(ps):
	a = [int(np.argmax(p)) for p in ps]
	return float(np.mean([a[i] == a[j] for i in range(len(a)) for j in range(i + 1, len(a))]))


def bias_and_null(fam_a, fam_b):
	"""JS between the two family means, and what it would be if both families
	were the same distribution: JS between half-means scales with the variance
	of a mean of K/2 draws, twice that of a mean of K, hence the 1/2."""
	h = len(fam_a) // 2
	half = [js(np.mean(f[:h], axis=0), np.mean(f[h:], axis=0)) for f in (fam_a, fam_b)]
	ma, mb = np.mean(fam_a, axis=0), np.mean(fam_b, axis=0)
	return js(ma, mb), 0.5 * float(np.mean(half)), int(np.argmax(ma) != np.argmax(mb))


def _norm(p):
	p = np.asarray(p, dtype=np.float64)
	return p / p.sum()


# ---------------------------------------------------------------- searching
class _ObservedNet:
	"""The network queried on the mover's OBSERVATION, which is what Coach
	stores as training input. Own Game instance: never touches the searcher's board."""
	def __init__(self, net, game):
		self.net, self.game = net, game

	def predict(self, board, valid_actions):
		return self.net.predict(np.array(self.game.getObservation(board, 0), copy=True), valid_actions)


class Prober:
	def __init__(self, Game, net, args, hidden):
		self.game, self.hidden, self.sims = Game(), hidden, args.numMCTSSims
		self.mcts = MCTS(self.game, net, args)
		self.mcts_obs = MCTS(self.game, _ObservedNet(net, Game()), args) if hidden else None
		self.cache = {}

	def search(self, obs, world_seed, chance_seed, dirichlet=None, observed=False, sampled=None):
		"""One full search from a fresh tree, as MCTS.getActionProb runs it for a
		hidden-info game at -u 1, with the world seed and the chance seed set
		separately. Returns (visit distribution = the training target at temp 1,
		root Q vector, root valid moves), or None if the root is terminal."""
		key = (world_seed, chance_seed, dirichlet, observed, sampled)
		if key in self.cache:
			return self.cache[key]
		mcts = self.mcts_obs if observed else self.mcts
		mcts.nodes_data = {}
		if dirichlet is not None:
			mcts.rng = np.random.default_rng(dirichlet)
		root = np.array(self.game.sampleWorld(obs, world_seed) if self.hidden else obs, copy=True)
		for step in range(self.sims):
			mcts.step = step
			mcts.random_seed = (chance_seed if sampled is None
			                    else 1 + hashed_draw(7919 * (sampled + 1), step, 2147483646))
			mcts.search(root, dirichlet_noise=(dirichlet is not None and step == 0),
			            forced_playouts=False, is_root=True, depth=0)
		node = mcts.nodes_data.get(self.game.stringRepresentation(root))
		res = None
		if node is not None and node[3] is not None:
			res = (_norm(node[5]), np.array(node[3][1], dtype=np.float64), np.asarray(node[1]).astype(bool))
		self.cache[key] = res
		return res


def phase_of(game, board):
	try:
		from catan.CatanConstants import GA_PHASE
		game.board.copy_state(np.array(board, copy=True), False)
		return int(game.board.globals_[0, GA_PHASE])
	except Exception:
		return -1


# ---------------------------------------------------------------- main
def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('buffer')
	ap.add_argument('checkpoint')
	ap.add_argument('--game', default='catan')
	ap.add_argument('--positions', type=int, default=120)
	ap.add_argument('--k', type=int, default=6, help='re-runs per source of variation (even, >= 4)')
	ap.add_argument('--sims', type=int, default=None, help='default: numMCTSSims stored in the checkpoint')
	ap.add_argument('--iter', type=int, default=-1, help='which iteration of the buffer to draw positions from')
	ap.add_argument('--seed', type=int, default=0, help='position sampling')
	ap.add_argument('--out', default=None, help='per-position metrics as JSON')
	args = ap.parse_args()
	if args.k < 4 or args.k % 2:
		sys.exit('--k must be even and >= 4 (the null of the bias statistics splits each family in halves)')
	K = args.k

	Game, NNet, _, _ = import_game(args.game)
	game = Game()
	hidden = hasattr(game, 'getObservation')
	net, ck = load_net(game, NNet, args.checkpoint)
	sargs = search_args(ck, args.sims)
	q_weight = float(ck.get('q_weight', 0.5) or 0.5)
	print(f'{args.checkpoint}: V{ck["nn_version"]}; search profile: sims={sargs.numMCTSSims} cpuct={sargs.cpuct} '
	      f'fpu={sargs.fpu} fpu_root={sargs.fpu_root} dirichlet={sargs.dirichletAlpha} '
	      f'softmax_temp={sargs.temperature[2]}; q_weight={q_weight}; hidden info: {hidden}')

	examples = load_iteration(args.buffer, args.iter)
	rng = np.random.default_rng(args.seed)
	candidates = [i for i, e in enumerate(examples) if int(np.asarray(e[3]).sum()) >= 2]
	picked = rng.choice(candidates, size=min(args.positions, len(candidates)), replace=False)
	print(f'{len(picked)} positions drawn among {len(candidates)} with >= 2 legal moves\n')

	seeds = [TRAINING_SEED] + [1 + hashed_draw(20260923, i, 2147483646) for i in range(1, K)]
	prober = Prober(Game, net, sargs, hidden)
	fam_names = ['universe', 'chance', 'hands', 'dirichlet', 'sampled', 'observed'] if hidden else \
		['chance', 'dirichlet', 'sampled']
	rec, n_not_idem, n_valid_mismatch, n_skipped, t0 = [], 0, 0, 0, time.perf_counter()

	for n, idx in enumerate(picked):
		board, pi_stored, _, valids, _ = examples[idx]
		obs = np.array(board, copy=True)
		valids = np.asarray(valids).astype(bool)
		if hidden and not np.array_equal(np.asarray(game.getObservation(obs, 0)), obs):
			n_not_idem += 1
		prober.cache = {}
		s0 = seeds[0]
		fam = {
			'chance': [prober.search(obs, s0, s) for s in seeds],
			'dirichlet': [prober.search(obs, s0, s0, dirichlet=1000 + i) for i in range(K)],
			'sampled': [prober.search(obs, s0, None, sampled=i) for i in range(K)],
		}
		if hidden:
			fam['universe'] = [prober.search(obs, s, s) for s in seeds]
			fam['hands'] = [prober.search(obs, s, s0) for s in seeds]
			fam['observed'] = [prober.search(obs, s, s, observed=True) for s in seeds]
		if any(r is None for f in fam.values() for r in f):
			n_skipped += 1
			continue
		base_valids = fam['chance'][0][2]
		if not np.array_equal(base_valids, valids):
			n_valid_mismatch += 1

		pis = {k: [r[0] for r in v] for k, v in fam.items()}
		qs = {k: np.array([r[1] for r in v]) for k, v in fam.items()}
		r = dict(index=int(idx), phase=phase_of(game, obs), n_legal=int(valids.sum()))
		for k in fam:
			r[f'disp_{k}'] = dispersion(pis[k])
			r[f'top1_{k}'] = top1_agreement(pis[k])
			r[f'sdq_{k}'] = float(np.std(qs[k][:, 0], ddof=1))
			r[f'varq_{k}'] = float(np.mean(np.var(qs[k], axis=0, ddof=1)))
		r['bias_sampled'], r['null_sampled'], r['top1diff_sampled'] = bias_and_null(pis['chance'], pis['sampled'])
		r['dq_sampled'] = float(qs['chance'][:, 0].mean() - qs['sampled'][:, 0].mean())

		# the network alone: what training fits (observation) vs what the search queries (world)
		p_obs, v_obs = net.predict(obs, valids)
		p_obs, v_obs = _norm(p_obs), np.asarray(v_obs, dtype=np.float64)
		ref = pis['universe'] if hidden else pis['chance']
		r['loss_stored'] = kl(_norm(pi_stored), p_obs)
		r['loss_universe'] = float(np.mean([kl(p, p_obs) for p in ref]))
		if hidden:
			r['bias_observed'], r['null_observed'], r['top1diff_observed'] = bias_and_null(pis['universe'], pis['observed'])
			r['dq_observed'] = float(qs['universe'][:, 0].mean() - qs['observed'][:, 0].mean())
			outs = [net.predict(np.array(game.sampleWorld(obs, s), copy=True), valids) for s in seeds]
			p_w = [_norm(o[0]) for o in outs]
			v_w = np.array([np.asarray(o[1], dtype=np.float64)[0] for o in outs])
			r['net_js_obs_world'] = float(np.mean([js(p_obs, p) for p in p_w]))
			r['net_spread_worlds'] = dispersion(p_w)
			r['net_top1_same'] = float(np.mean([np.argmax(p) == np.argmax(p_obs) for p in p_w]))
			r['net_dv'] = float(np.mean(np.abs(v_w - v_obs[0])))
			r['net_sdv_worlds'] = float(np.std(v_w, ddof=1))
		rec.append(r)

		if n in (0, 4) or (n + 1) % 20 == 0:
			el = time.perf_counter() - t0
			print(f'  {n + 1}/{len(picked)} positions, {el / 60:.1f} min, ~{el / (n + 1) * (len(picked) - n - 1) / 60:.0f} min left')

	if not rec:
		sys.exit('no usable position')
	report(rec, fam_names, hidden, K, q_weight, n_not_idem, n_valid_mismatch, n_skipped)
	if args.out:
		with open(args.out, 'w') as f:
			json.dump(dict(args=vars(args), seeds=[int(s) for s in seeds], positions=rec), f)
		print(f'\nper-position metrics written to {args.out}')


def _ci(xs):
	xs = np.asarray(xs, dtype=np.float64)
	m = float(xs.mean())
	h = 1.96 * float(xs.std(ddof=1)) / np.sqrt(len(xs)) if len(xs) > 1 else float('nan')
	return m, h


def report(rec, fam_names, hidden, K, q_weight, n_not_idem, n_valid_mismatch, n_skipped):
	n = len(rec)
	col = lambda key: [r[key] for r in rec]
	print(f'\n{n} positions, K={K} re-runs per source'
	      + (f'; {n_skipped} skipped (terminal root)' if n_skipped else ''))
	if n_not_idem:
		print(f'  WARNING: getObservation is not idempotent on {n_not_idem} stored positions')
	if n_valid_mismatch:
		print(f'  WARNING: root legal moves differ from the stored ones on {n_valid_mismatch} positions')

	loss_st, h_st = _ci(col('loss_stored'))
	loss_u, h_u = _ci(col('loss_universe'))
	print(f'\npolicy loss of the net on these positions: stored targets {loss_st:.4f} ±{h_st:.4f}, '
	      f'noise-free single-universe targets {loss_u:.4f} ±{h_u:.4f}')

	w = 44
	print(f'\n{"source of variation":<{w}s} {"dispersion":>17s} {"top-1 agree":>11s} {"sd Q(mover)":>11s} {"Q-part of value MSE":>19s}')
	labels = {'universe': 'universe = chance + hands (what -u 1 fixes)', 'chance': '  chance only (dice, dev draws, steals)',
	          'hands': '  invented opponent hands only', 'dirichlet': 'Dirichlet, same universe (for scale)',
	          'sampled': 'sampled chance, one tree (MCTS noise)', 'observed': 'universe, net queried on observation'}
	c2 = (q_weight / (1 + q_weight)) ** 2
	for k in fam_names:
		d, h = _ci(col(f'disp_{k}'))
		print(f'{labels[k]:<{w}s} {d:9.4f} ±{h:.4f} {np.mean(col(f"top1_{k}")):10.0%} '
		      f'{np.mean(col(f"sdq_{k}")):11.3f} {c2 * np.mean(col(f"varq_{k}")):19.4f}')

	main_src = 'universe' if hidden else 'chance'
	share = np.mean(col(f'disp_{main_src}')) / loss_st
	share_dir = np.mean(col('disp_dirichlet')) / loss_st
	print(f'\nshare of the stored-target policy loss that is {main_src} noise: {share:.0%}'
	      f'   (Dirichlet noise, intended: {share_dir:.0%})')
	print(f'  (the Q-part column is ({q_weight:g}/{1 + q_weight:g})^2 x var(Q): the value-MSE floor that the '
	      f'universe alone puts in the q-mixed target)')

	b, hb = _ci(col('bias_sampled'))
	print(f'\nclairvoyance bias: JS(mean of fixed-chance targets, mean of sampled-chance targets) = {b:.4f} ±{hb:.4f}'
	      f'   null ~{np.mean(col("null_sampled")):.4f}')
	dq, hq = _ci(col('dq_sampled'))
	print(f'  best move changes on {np.mean(col("top1diff_sampled")):.0%} of positions;'
	      f' root Q(mover) fixed minus sampled = {dq:+.3f} ±{hq:.3f}')
	if hidden:
		b, hb = _ci(col('bias_observed'))
		print(f'\ninput coherence, search level: JS(mean current target, mean target with the net on observations)'
		      f' = {b:.4f} ±{hb:.4f}   null ~{np.mean(col("null_observed")):.4f}')
		dq, hq = _ci(col('dq_observed'))
		print(f'  best move changes on {np.mean(col("top1diff_observed")):.0%} of positions;'
		      f' root Q(mover) current minus observed = {dq:+.3f} ±{hq:.3f}')
		j, hj = _ci(col('net_js_obs_world'))
		print(f'input coherence, network level: JS(net on observation, net on an invented world) = {j:.4f} ±{hj:.4f}'
		      f'   (spread between worlds {np.mean(col("net_spread_worlds")):.4f})')
		print(f'  same top move on {np.mean(col("net_top1_same")):.0%} of (position, world) pairs;'
		      f' |v(world) - v(observation)| = {np.mean(col("net_dv")):.3f}'
		      f'   (sd of v between worlds {np.mean(col("net_sdv_worlds")):.3f})')

	groups = {}
	for r in rec:
		groups.setdefault(PHASE_GROUPS.get(r['phase'], f'phase {r["phase"]}'), []).append(r)
	print(f'\nby phase ({main_src} dispersion | clairvoyance bias | top move changed by sampling):')
	for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
		print(f'  {g:<16s} n={len(rs):4d}   {np.mean([r[f"disp_{main_src}"] for r in rs]):.4f}   '
		      f'{np.mean([r["bias_sampled"] for r in rs]):.4f}   {np.mean([r["top1diff_sampled"] for r in rs]):.0%}')

	print('\nReading (thresholds declared before the run):')
	print(f'  - {main_src} share >= ~50%: the target, not the network, sets the policy floor; a search that does not')
	print('    bet on one universe is the lever. <= ~15%: the universe is not what limits the policy.')
	print('  - clairvoyance bias well above its null AND best move changed on a sizable share: averaging clairvoyant')
	print('    targets is not planning under uncertainty; sampled chance nodes would teach different moves.')
	if hidden:
		print('  - input coherence: net(observation) far from net(world) relative to the spread between worlds, or the')
		print('    observed-input search moving the target beyond its null => the searched function is not the')
		print('    trained one. Fix that before reading anything else.')


if __name__ == '__main__':
	main()
