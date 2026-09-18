"""Assertions on CatanNNet. Run before the first training run.

The one that matters most is EQUIVARIANCE. get_symmetries() multiplies every
sample by 12; if the network does not satisfy f(sigma . s) == sigma . f(s), the
augmented (state, policy) pairs are mutually inconsistent and the augmentation
is worse than useless -- it teaches the network that the same position has 12
different answers. The failure is silent: training simply converges to a worse
network, with nothing in the logs to point at.

Usage:
    python CatanNNetTest.py            # add --flops for the FLOP count
"""
import sys

import numpy as np
import torch

try:
	from CatanConstants import *
	from CatanNNet import CatanNNet, VERSIONS, N_TOKENS
	from CatanTest import build_reference_state, permute_state, check_state
except ImportError:                                  # pragma: no cover
	from .CatanConstants import *
	from .CatanNNet import CatanNNet, VERSIONS, N_TOKENS
	from .CatanTest import build_reference_state, permute_state, check_state


class _FakeGame:
	num_players = N_PLAYERS


def _net(version, seed=0):
	torch.manual_seed(seed)
	net = CatanNNet(_FakeGame(), version)
	net.eval()
	return net


def check_nn_args_dict(version):
	"""The real call site is catan/NNet.py: nn_model(game, nn_args), with nn_args
	the FULL dict built in main.py -- not just the version number. This is
	exactly the call that crashed with TypeError: unhashable type: 'dict'."""
	nn_args = dict(lr=1e-3, dropout=0., epochs=2, batch_size=32, nn_version=version,
	              learn_rate=1e-3, no_compression=False, q_weight=0.5)
	net = CatanNNet(_FakeGame(), nn_args)
	assert net.version == version
	bad = dict(nn_args, nn_version=999)
	try:
		CatanNNet(_FakeGame(), bad)
		raise AssertionError('an unsupported nn_version must raise, not build a broken net')
	except ValueError:
		pass
	print(f'  V{version} nn_args dict construction      OK')


def _batch(seeds):
	states = []
	for s in seeds:
		st = build_reference_state(seed=s, n_settlements=2)
		check_state(st)
		states.append(st)
	return torch.from_numpy(np.stack(states).astype(np.float32))


def check_shapes(version):
	net = _net(version)
	board = _batch(range(4))
	valids = torch.ones(4, N_ACTIONS, dtype=torch.bool)
	# The trade block is masked here only to have a KNOWN-invalid range to test
	# masking against; it is a real block now, not a reserved one.
	valids[:, N_ACTIONS_V1:] = False
	pi, v = net(board, valids)
	assert pi.shape == (4, N_ACTIONS), pi.shape
	assert v.shape == (4, N_PLAYERS), v.shape
	assert torch.allclose(pi.exp().sum(-1), torch.ones(4), atol=1e-5), 'pi is not a distribution'
	assert (v.abs() <= 1).all(), 'value head escaped tanh'
	# masked actions must receive essentially zero probability
	assert pi.exp()[:, N_ACTIONS_V1:].max() < 1e-8, 'invalid actions carry probability'
	assert torch.isfinite(pi).all() and torch.isfinite(v).all()
	print(f'  V{version} shapes / masking / normalisation  OK')


def check_equivariance(version, tol=2e-4):
	"""f(sigma . s) must equal sigma . f(s) for the 12 isometries."""
	net = _net(version)
	worst = 0.
	for seed in range(3):
		st = build_reference_state(seed=seed, n_settlements=2)
		valids = np.ones(N_ACTIONS, dtype=bool)
		valids[N_ACTIONS_V1:] = False
		b0 = torch.from_numpy(st[None].astype(np.float32))
		v0 = torch.from_numpy(valids[None])
		with torch.no_grad():
			pi0, val0 = net(b0, v0)
		for s in range(N_ISOMETRIES):
			st_s = permute_state(st, s)
			perm = ISO_ACTION[s].astype(np.int64)
			val_s = np.empty_like(valids)
			val_s[perm] = valids
			with torch.no_grad():
				pi_s, val_s_out = net(torch.from_numpy(st_s[None].astype(np.float32)),
				                      torch.from_numpy(val_s[None]))
			ref = torch.empty_like(pi0)
			ref[:, torch.from_numpy(perm)] = pi0
			worst = max(worst, float((pi_s - ref).abs().max()), float((val_s_out - val0).abs().max()))
			assert torch.allclose(pi_s, ref, atol=tol), \
				f'policy not equivariant under isometry {s}: max diff {float((pi_s - ref).abs().max()):.2e}'
			assert torch.allclose(val_s_out, val0, atol=tol), \
				f'value not invariant under isometry {s}'
	print(f'  V{version} equivariance over 12 isometries  OK  (worst deviation {worst:.1e})')


def check_edge_head(version):
	"""The edge head has no token of its own: prove it can still tell edges apart."""
	net = _net(version)
	board = _batch(range(6))
	valids = torch.ones(6, N_ACTIONS, dtype=torch.bool)
	with torch.no_grad():
		pi, _ = net(board, valids)
	logits = pi[:, A_ROAD:A_ROAD + N_EDGES]
	spread = logits.std(dim=-1)
	assert (spread > 1e-3).all(), f'edge logits collapsed to a constant (std {spread.min():.2e})'
	# two edges sharing a vertex must be separable, otherwise the bilinear form
	# has degenerated into a per-vertex score
	pairs = [(int(VERTEX_TO_EDGE[v, 0]), int(VERTEX_TO_EDGE[v, 1]))
	         for v in range(N_VERTICES) if VERTEX_TO_EDGE[v, 1] != NO_EDGE]
	diffs = torch.stack([(logits[:, a] - logits[:, b]).abs().max() for a, b in pairs])
	assert diffs.max() > 1e-3, 'adjacent edges always share the same logit'
	assert (diffs > 1e-4).float().mean() > .5, 'most adjacent edge pairs are indistinguishable'
	print(f'  V{version} edge head discriminates       OK  (logit std {float(spread.mean()):.2f})')


def check_masking_flag(version):
	"""The `masked` input must fire exactly when a hand has been hidden."""
	net = _net(version)
	st = build_reference_state(seed=11, n_settlements=2)
	board = torch.from_numpy(st[None].astype(np.float32))
	with torch.no_grad():
		h_clear = net._encode(board)
	masked = st.copy()
	for p in range(1, N_PLAYERS):
		masked[ROW_PLAYER + 4 * p, PA_RESOURCES:PA_RESOURCES + N_RESOURCES] = 0
	with torch.no_grad():
		h_masked = net._encode(torch.from_numpy(masked[None].astype(np.float32)))
	rows = [TOK_PLAYER_ROW + p for p in range(1, N_PLAYERS)]
	assert (h_masked[0, rows] - h_clear[0, rows]).abs().max() > 1e-3, \
		'masking an opponent hand does not change its token'
	assert (h_masked[0, TOK_PLAYER_ROW] - h_clear[0, TOK_PLAYER_ROW]).abs().max() < 1e-6, \
		"masking an opponent changed the viewer's own token"
	print(f'  V{version} masking flag                   OK')


def check_trade_offer_is_encoded(version):
	"""A standing offer must reach the network, and reach it on ITS OWN AUTHOR's
	token. The net answers OK/NO and counters, so an offer it cannot see makes
	those three phases blind guesses -- and an offer it sees but cannot attribute
	makes A_TRADE_ACCEPT + t undecidable."""
	net = _net(version)
	st = build_reference_state(seed=5, n_settlements=2)
	board = torch.from_numpy(st[None].astype(np.float32))
	with torch.no_grad():
		h0 = net._encode(board)
	for p in range(N_PLAYERS):
		st_p = st.copy()
		d = ROW_PLAYER + ROWS_PER_PLAYER * p + 3
		st_p[d, PD_TRADE_RECV + ORE] = 1
		st_p[d, PD_TRADE_GIVE + LUMBER] = 2
		st_p[d, PD_TRADE_STATUS] = TRADE_OFFERED
		with torch.no_grad():
			h1 = net._encode(torch.from_numpy(st_p[None].astype(np.float32)))
		moved = (h1[0] - h0[0]).abs().amax(dim=-1)
		assert moved[TOK_PLAYER_ROW + p] > 1e-3, f"p{p}'s offer never reaches its own token"
		for q in range(N_PLAYERS):
			if q != p:
				assert moved[TOK_PLAYER_ROW + q] < 1e-6, \
					f"p{p}'s offer leaked into p{q}'s token: the author is ambiguous"
	print(f'  V{version} trade offer encoded per author OK')


TOK_PLAYER_ROW = N_VERTICES + N_HEXES


def check_onnx(version):
	"""The inference path is ONNX, so an export failure means a dead network."""
	import os, tempfile
	try:
		import onnxruntime as ort
	except ImportError:
		print('  onnxruntime not installed, skipping the export check')
		return
	net = _net(version)
	board = _batch([0])
	valids = torch.ones(1, N_ACTIONS, dtype=torch.bool)
	path = os.path.join(tempfile.gettempdir(), f'catan_v{version}_test.onnx')
	torch.onnx.export(net, (board, valids), path,
	                  input_names=['board', 'valid_actions'], output_names=['pi', 'v'],
	                  dynamic_axes={'board': {0: 'b'}, 'valid_actions': {0: 'b'},
	                                'pi': {0: 'b'}, 'v': {0: 'b'}},
	                  opset_version=17, dynamo=False)
	sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
	big = _batch(range(5)).numpy()
	vb = np.ones((5, N_ACTIONS), dtype=bool)
	out = sess.run(None, {'board': big, 'valid_actions': vb})
	with torch.no_grad():
		pi, v = net(torch.from_numpy(big), torch.from_numpy(vb))
	assert np.abs(out[0] - pi.numpy()).max() < 1e-4, 'ONNX and torch disagree on pi'
	assert np.abs(out[1] - v.numpy()).max() < 1e-4, 'ONNX and torch disagree on v'
	os.remove(path)
	print(f'  V{version} ONNX export + dynamic batch  OK')


def check_dummy_trace_robustness(version):
	"""export_and_load_onnx() (and FlopCountAnalysis, and torch.jit tracing in
	general) calls forward() on a DUMMY board of unconstrained torch.randn
	floats, purely to capture shapes. This crashed in practice with
	'IndexError: index out of range in self' the first time it ran for real,
	because embedding lookups read raw board columns with no bound. Every
	board-derived embedding index must tolerate garbage input."""
	net = _net(version)
	for _ in range(20):
		scale = 10 ** np.random.randint(0, 4)
		board = torch.randn(2, N_ROWS, N_COLS) * scale
		valids = torch.rand(2, N_ACTIONS) > 0.5
		pi, v = net(board, valids)
		assert torch.isfinite(pi).all() and torch.isfinite(v).all(), 'non-finite output on garbage input'
	print(f'  V{version} tolerates unconstrained dummy input (ONNX/FX tracing)  OK')


def check_flops():
	try:
		from fvcore.nn import FlopCountAnalysis
	except ImportError:
		print('  fvcore not installed, skipping the FLOP count')
		return
	board = _batch([0])
	valids = torch.ones(1, N_ACTIONS, dtype=torch.bool)
	for version in sorted(VERSIONS):
		net = _net(version)
		f = FlopCountAnalysis(net, (board, valids))
		f.unsupported_ops_warnings(False)
		f.uncalled_modules_warnings(False)
		params = sum(p.numel() for p in net.parameters())
		# fvcore counts MACs; the usual convention is 2 FLOPs per MAC
		print(f'  V{version:2d} {VERSIONS[version]["name"]:5s} d={VERSIONS[version]["dim"]:2d} '
		      f'L={VERSIONS[version]["layers"]}  {2 * f.total() / 1e6:5.2f} MFlops '
		      f'({f.total() / 1e6:5.2f} MMac)  {params / 1e3:5.1f}k params')


if __name__ == '__main__':
	for version in sorted(VERSIONS):
		check_shapes(version)
		check_equivariance(version)
		check_edge_head(version)
		check_masking_flag(version)
		check_trade_offer_is_encoded(version)
		check_onnx(version)
		check_dummy_trace_robustness(version)
		check_nn_args_dict(version)
	print()
	check_flops()
	print('\nCATAN NNET TEST BATTERY: DONE')
