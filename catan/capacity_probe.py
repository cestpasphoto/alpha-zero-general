"""Offline capacity probe: does a BIGGER net fit the SAME data better?

Answers, without spending a from-scratch self-play run, the only question a new
architecture raises: is the network the bottleneck? Each version is trained from
scratch on the same replay buffer, with the same budget, and scored on held-out
positions (the most recent iteration, which no version has trained on).

    python catan/capacity_probe.py <path to checkpoint.examples> \
        [--versions 13,16] [--lrs 3e-4,1e-3] [--steps 3000] [--batch 32] \
        [--q-weight 0.5] [--every 250] [--val 20000]

Reading it:
  - a bigger net that reaches a clearly LOWER validation loss => capacity is the
    limiting factor, a from-scratch run is worth its cost;
  - same validation loss => capacity is not the bottleneck, and the extra
    parameters will only cost self-play time.
Losses are the training ones (KLDiv on the policy, MSE on the V/Q mix), so they
are directly comparable to what the training loop prints.

WARM-START mode (--init): the from-scratch mode above says nothing about the
real pipeline, which starts every iteration from the previous network. This
mode replays ONE training step of the pipeline, exactly as GenericNNetWrapper
.train does it (fresh AdamW, OneCycleLR(max_lr=lr, epochs=p), batches drawn
without replacement), from a real checkpoint, for several (lr, epochs)
schedules, and scores each result on the held-out iteration:

    python catan/capacity_probe.py <checkpoint.examples> --init <checkpoint_K.pt> \
        [--schedules 3e-4:2,1e-3:8,3e-3:8] [--seeds 0,1,2]

--init and the buffer must come from the SAME run, at the SAME moment. The
buffer holds iterations N-4..N (written after the self-play of N); take the last
ACCEPTED checkpoint K <= N-2, so that only iteration N-1 is new to the network,
as in the pipeline. Never K = N (it has trained on the held-out iteration), and
never a checkpoint much older than the buffer: a stale network has seen none of
the training iterations, which turns the replay into a different regime that
favours longer / stronger schedules.

--seeds repeats every schedule with each seed (batch order and init of the
RNGs); the summary then gives mean +- half-range. A difference smaller than
that spread is noise. The summary scores policy and value SEPARATELY: the
total weighs value by 0.25, so a schedule can lower it while degrading the
value head.
"""
import os
import re
import argparse, pickle, sys, time, zlib
import numpy as np
import torch

sys.path.insert(0, '.')
try:
	from CatanConstants import N_PLAYERS
	from CatanNNet import CatanNNet, VERSIONS
except ImportError:                                  # pragma: no cover
	from catan.CatanConstants import N_PLAYERS
	from catan.CatanNNet import CatanNNet, VERSIONS


class _FakeGame:
	num_players = N_PLAYERS


def _is_example(x):
	"""An example is a zlib blob, or the raw 5-tuple (board, pi, v, valids, q) --
	as opposed to a CONTAINER of examples, which is what one iteration is."""
	return isinstance(x, (bytes, bytearray)) or (
		isinstance(x, tuple) and len(x) == 5 and hasattr(x[0], 'shape'))


def load_buffer(path):
	"""checkpoint.examples is a pickle of Coach.trainExamplesHistory: one entry per
	iteration -- a deque, list or tuple -- of examples, each either a raw 5-tuple
	or a zlib-compressed pickle of one (the two may be mixed, as loadTrainExamples
	harmonises them after loading). A flat list of examples is accepted too and cut
	into ten pseudo-iterations, so a single-iteration dump still works."""
	with open(path, 'rb') as f:
		raw = f.read()
	try:
		hist = pickle.loads(raw)
	except Exception:
		hist = pickle.loads(zlib.decompress(raw))        # whole-file compression
	try:
		hist = list(hist)
	except TypeError:
		sys.exit(f'unexpected buffer layout: {type(hist).__name__} is not iterable')
	if not hist:
		sys.exit('empty buffer')
	if _is_example(hist[0]):
		n = max(len(hist) // 10, 1)
		iters = [hist[i:i + n] for i in range(0, len(hist), n)]
		print('flat list of examples: cut into 10 pseudo-iterations')
	else:
		iters = [list(it) for it in hist]
		if iters and not _is_example(iters[0][0]):
			sys.exit(f'unexpected buffer layout: iteration items are {type(iters[0][0]).__name__}, '
			         f'expected zlib blobs or 5-tuples')
	print(f'{len(iters)} iterations, {sum(len(i) for i in iters)} examples '
	      f'({[len(i) for i in iters]})')
	return iters


def to_tensors(raw):
	ex = [pickle.loads(zlib.decompress(e)) if isinstance(e, (bytes, bytearray)) else e for e in raw]
	boards, pis, vs, valids, qs = list(zip(*ex))
	return (torch.FloatTensor(np.array(boards, dtype=np.float32)),
	        torch.BoolTensor(np.array(valids, dtype=np.bool_)),
	        torch.FloatTensor(np.array(pis, dtype=np.float32)),
	        torch.FloatTensor(np.array(vs, dtype=np.float32)),
	        torch.FloatTensor(np.array(qs, dtype=np.float32)))


def load_ckpt(path):
	"""torch.save'd dict from GenericNNetWrapper.save_checkpoint: state_dict,
	nn_version, a pickled full_model, and the run's CLI args. The full_model
	pickle may name a module path that does not import from here, so fall back to
	stubbing unknown classes -- only the state_dict and the args are used."""
	try:
		return torch.load(path, map_location='cpu', weights_only=False)
	except Exception:
		import types

		class _Unpickler(pickle.Unpickler):
			def find_class(self, mod, name):
				try:
					return super().find_class(mod, name)
				except Exception:
					return type(name, (), {'__setstate__': lambda self, st: None})
		stub = types.ModuleType('stub_pickle')
		stub.Unpickler, stub.load, stub.loads = _Unpickler, pickle.load, pickle.loads
		return torch.load(path, map_location='cpu', weights_only=False, pickle_module=stub)


def losses(net, b, va, tpi, tv, tq, q_weight):
	out_pi, out_v = net(b, va)
	l_pi = torch.nn.KLDivLoss(reduction='batchmean')(out_pi, tpi)
	tgt = (tv + q_weight * tq) / (1 + q_weight)
	l_v = torch.sum((tgt - out_v) ** 2) / (tv.size(0) * tv.size(-1))
	return l_pi, l_v


def _full_loss(net, b, va, tpi, tv, tq, q_weight):
	acc_pi = acc_v = 0.0
	for s in range(0, len(b), 1024):
		p, v = losses(net, b[s:s+1024], va[s:s+1024], tpi[s:s+1024], tv[s:s+1024], tq[s:s+1024], q_weight)
		n = len(b[s:s+1024]); acc_pi += float(p) * n; acc_v += float(v) * n
	return acc_pi / len(b), acc_v / len(b)


def run(version, lr, train, val, probe, args, steps=None, init_state=None, every=None, seed=0):
	torch.manual_seed(seed); np.random.seed(seed)
	net = CatanNNet(_FakeGame(), version)
	if init_state is not None:
		net.load_state_dict(init_state)
	steps, every = steps or args.steps, every or args.every
	n_par = sum(p.numel() for p in net.parameters())
	opt = torch.optim.AdamW(net.parameters(), lr=lr)
	sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
	vb, vva, vpi, vv, vq = val
	tb, tva, tpi_, tv_, tq_ = probe                    # fixed slice of the TRAINING set
	best, hist, t0 = float('inf'), [], time.perf_counter()
	for step in range(steps):
		net.train()
		ids = np.random.choice(len(train), size=args.batch, replace=False)
		b, va, tpi, tv, tq = to_tensors([train[i] for i in ids])
		opt.zero_grad(set_to_none=True)
		l_pi, l_v = losses(net, b, va, tpi, tv, tq, args.q_weight)
		(l_pi + 0.25 * l_v).backward()
		opt.step(); sched.step()
		if (step + 1) % every == 0 or step == steps - 1:
			net.eval()
			with torch.no_grad():
				(vpi_l, vv_l), (tpi_l, tv_l) = (_full_loss(net, *s_, args.q_weight)
				                                for s_ in ((vb, vva, vpi, vv, vq), (tb, tva, tpi_, tv_, tq_)))
			hist.append((step + 1, vpi_l, vv_l, tpi_l, tv_l))
			best = min(best, vpi_l + 0.25 * vv_l)
			print(f'    step {step+1:6d}  val {vpi_l:.4f}/{vv_l:.4f} = {vpi_l + 0.25*vv_l:.4f}   '
			      f'train {tpi_l:.4f}/{tv_l:.4f} = {tpi_l + 0.25*tv_l:.4f}   '
			      f'gap {(vpi_l + 0.25*vv_l) - (tpi_l + 0.25*tv_l):+.4f}')
	return dict(version=version, lr=lr, params=n_par, best=best, hist=hist,
	            gap=hist[-1][1] + 0.25*hist[-1][2] - hist[-1][3] - 0.25*hist[-1][4],
	            minutes=(time.perf_counter() - t0) / 60)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('buffer')
	ap.add_argument('--versions', default='13,16')
	ap.add_argument('--lrs', default='3e-4')
	ap.add_argument('--steps', type=int, default=3000)
	ap.add_argument('--batch', type=int, default=32)
	ap.add_argument('--every', type=int, default=250)
	ap.add_argument('--val', type=int, default=20000, help='held-out examples taken from the LAST iteration')
	ap.add_argument('--q-weight', type=float, default=None, help='default: the checkpoint\'s, else 0.5')
	ap.add_argument('--init', default=None, help='warm-start mode: a real checkpoint_N.pt (see the docstring)')
	ap.add_argument('--schedules', default='3e-4:2,1e-3:8,3e-3:8', help='warm-start mode: lr:epochs,...')
	ap.add_argument('--seeds', default='0', help='warm-start mode: repeat each schedule with these seeds')
	args = ap.parse_args()

	iters = load_buffer(args.buffer)
	if len(iters) < 2:
		sys.exit('need at least 2 iterations: the last one is the held-out set')
	val_raw = iters[-1][:args.val]
	train = [e for it in iters[:-1] for e in it]
	print(f'train {len(train)} examples (iterations 1..{len(iters)-1}), '
	      f'validation {len(val_raw)} examples (iteration {len(iters)}, never trained on)\n')
	val = to_tensors(val_raw)
	# A fixed slice of the training set, scored the same way: val alone cannot say
	# whether a net is short of capacity (train ~= val, both high) or short of data
	# (train << val).
	rng = np.random.default_rng(0)
	probe = to_tensors([train[i] for i in rng.choice(len(train), size=min(args.val, len(train)), replace=False)])

	if args.init:
		return warm_start(args, train, val, probe)
	args.q_weight = 0.5 if args.q_weight is None else args.q_weight

	out = []
	for version in [int(v) for v in args.versions.split(',')]:
		cfg = VERSIONS[version]
		for lr in [float(x) for x in args.lrs.split(',')]:
			print(f'  V{version} {cfg["name"]} d={cfg["dim"]} L={cfg["layers"]}  lr={lr:g}')
			out.append(run(version, lr, train, val, probe, args))
	print(f'\n{"version":>8s} {"params":>8s} {"lr":>8s} {"best val loss":>14s} {"val-train":>10s} {"minutes":>8s}')
	for r in out:
		print(f'{"V"+str(r["version"]):>8s} {r["params"]/1000:7.1f}k {r["lr"]:8.1e} {r["best"]:14.4f} {r["gap"]:+10.4f} {r["minutes"]:8.1f}')
	by_v = {}
	for r in out:
		by_v[r['version']] = min(by_v.get(r['version'], float('inf')), r['best'])
	ref = by_v[min(by_v)]
	print()
	for v in sorted(by_v):
		print(f'  V{v}: {by_v[v]:.4f}  ({(by_v[v] - ref) / ref * 100:+.1f}% vs V{min(by_v)})')
	print('\nA gap of a few percent is noise; a bigger net is worth a run only if it is clearly lower.')
	print('val-train near 0 with both losses still falling = short of capacity or of steps (raise --steps/--lrs);')
	print('val clearly above train = short of DATA, and more parameters will only make it worse.')


def _check_same_run(buffer_path, init_path):
	"""Warn when --init does not look like the right checkpoint for this buffer.
	Only accepted iterations leave a checkpoint_i.pt, and the right one is the
	last accepted K <= N-2, i.e. one of the newest few next to the buffer."""
	bdir, idir = (os.path.dirname(os.path.abspath(p)) for p in (buffer_path, init_path))
	m = re.search(r'checkpoint_(\d+)\.pt$', os.path.basename(init_path))
	k = int(m.group(1)) if m else None
	near = sorted(int(g.group(1)) for g in (re.search(r'^checkpoint_(\d+)\.pt$', f) for f in os.listdir(bdir)) if g)
	if idir != bdir:
		newest = f'newest there: checkpoint_{near[-1]}.pt' if near else 'no checkpoint_i.pt there'
		print(f'  WARNING: --init is not in the buffer\'s folder ({newest}). Unless the buffer was taken from\n'
		      f'  the run of --init while checkpoint_{k if k is not None else "K"} was one of its newest, this network is\n'
		      f'  stale or foreign to these data and the replay does NOT reproduce a pipeline step.\n')
	elif k is not None and len(near) > 3 and k < near[-3]:
		print(f'  WARNING: checkpoint_{k}.pt is older than the 3 newest checkpoints next to the buffer '
		      f'({near[-3:]}):\n  it has probably not trained on any of the training iterations '
		      f'(stale start, see the docstring).\n')


def _pm(xs, fmt='{:.4f}'):
	"""mean +- half-range over seeds (no spread with a single seed)"""
	m = sum(xs) / len(xs)
	return fmt.format(m) + (f' ±{(max(xs) - min(xs)) / 2:.4f}' if len(xs) > 1 else '        ')


def warm_start(args, train, val, probe):
	ck = load_ckpt(args.init)
	state = ck['state_dict']
	version = int(ck.get('nn_version', getattr(ck.get('full_model'), 'version', 0)) or 0)
	if version not in VERSIONS:
		sys.exit(f'cannot tell the network version of {args.init} (nn_version={ck.get("nn_version")})')
	keys = ('learn_rate', 'epochs', 'batch_size', 'q_weight', 'numItersHistory', 'numEps',
	        'numMCTSSims', 'updateThreshold', 'arenaCompare')
	print(f'{args.init}: V{version}; run settings stored in it: '
	      + ', '.join(f'{k}={ck[k]}' for k in keys if k in ck))
	_check_same_run(args.buffer, args.init)
	if args.q_weight is None:
		args.q_weight = float(ck.get('q_weight', 0.5))
	per_epoch = len(train) // args.batch             # GenericNNetWrapper.train: len(examples) / batch_size
	seeds = [int(s) for s in args.seeds.split(',')]

	net0 = CatanNNet(_FakeGame(), version)
	net0.load_state_dict(state)
	net0.eval()
	with torch.no_grad():
		(vpi0, vv0), (tpi0, tv0) = (_full_loss(net0, *s_, args.q_weight) for s_ in (val, probe))
	base_val, base_train = vpi0 + 0.25 * vv0, tpi0 + 0.25 * tv0
	print(f'\n  before any training: val {vpi0:.4f}/{vv0:.4f} = {base_val:.4f}   '
	      f'train {tpi0:.4f}/{tv0:.4f} = {base_train:.4f}\n')

	out = []
	for sch in args.schedules.split(','):
		lr, ep = sch.split(':')
		lr, ep = float(lr), int(ep)
		steps = ep * per_epoch
		runs = []
		for i, seed in enumerate(seeds):
			print(f'  -l {lr:g} -p {ep}: {steps} steps ({ep} passes over {len(train)} examples), seed {seed}')
			# the full curve for the first seed only: the others just need their end point
			runs.append(run(version, lr, train, val, probe, args, steps=steps, init_state=state,
			                every=max(steps // 4, 1) if i == 0 else steps, seed=seed))
		ends = [r['hist'][-1] for r in runs]         # (step, val_pi, val_v, train_pi, train_v)
		out.append(dict(lr=lr, epochs=ep, steps=steps,
		                pi=[e[1] for e in ends], v=[e[2] for e in ends],
		                tot=[e[1] + 0.25 * e[2] for e in ends],
		                gap=[r['gap'] for r in runs], minutes=sum(r['minutes'] for r in runs)))

	w = 15
	print(f'\n{"schedule":>16s} {"steps":>6s} {"val policy":>{w}s} {"val value":>{w}s} {"val total":>{w}s} '
	      f'{"vs before":>9s} {"val-train":>9s} {"minutes":>7s}')
	print(f'{"(no training)":>16s} {0:6d} {vpi0:{w}.4f} {vv0:{w}.4f} {base_val:{w}.4f} '
	      f'{0:+8.1f}% {base_val - base_train:+9.4f} {0:7.1f}')
	for r in out:
		tot = sum(r['tot']) / len(r['tot'])
		print(f'{"-l %g -p %d" % (r["lr"], r["epochs"]):>16s} {r["steps"]:6d} {_pm(r["pi"]):>{w}s} '
		      f'{_pm(r["v"]):>{w}s} {_pm(r["tot"]):>{w}s} {(tot - base_val) / base_val * 100:+8.1f}% '
		      f'{sum(r["gap"]) / len(r["gap"]):+9.4f} {r["minutes"]:7.1f}')
	if len(seeds) > 1:
		print(f'\nmean ± half-range over seeds {seeds}: a difference smaller than the spreads is noise.')
	print('\n"val total" is policy + 0.25 x value, the loss the pipeline minimises; it can go DOWN while the')
	print('value head gets WORSE, so read policy and value separately. A value ABOVE "no training" means')
	print('the schedule is undoing what earlier iterations taught the value head. Only a schedule that')
	print('lowers the policy without raising the value (beyond the spread) justifies changing -l / -p.')


if __name__ == '__main__':
	main()
