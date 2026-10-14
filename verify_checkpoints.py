#!/usr/bin/env python3
"""
verify_checkpoints.py -- read-only health check of ONE checkpoint file (.pt), new format.

  python verify_checkpoints.py [<game>] <ckpt.pt> [--complexity] [--onnx FILE]
                               [--onnx-opset 21,20,19,18,17,native] [--diff] [--restore DIR]
                               [--other-keys [main.py]]

Run from the framework root (next to GameSwitcher.py and GenericNNetWrapper.py). One line is printed per
test, as soon as it is done. Exit code: 0 no FAIL (WARN allowed), 2 FAIL or stopped run.
<game> is optional: it is read from provenance.game (pass it for a converted file, or to override
a renamed game folder: a different name then gives a WARN).
The file is loaded with torch.load(weights_only=True): plain types and tensors only, no game code
is needed to read it. An old-format file (pickled full_model) is refused: convert_checkpoint.py.

Standard keys (see GenericNNetWrapper.py)                                   status if wrong
  state_dict                 non-empty dict of tensors                FAIL
  provenance.nn_args         dict with int nn_version >= 1            FAIL
  provenance.board_size      list of positive ints                    FAIL
  provenance.action_size     int >= 1                                 FAIL
  provenance.num_players     int >= 2                                 FAIL
  provenance.saved_at        shown ('-' if absent), never a warning   ignored
  provenance.game            name for GameSwitcher; absent -> <game> required   FAIL if needed
  provenance.git/command_line/torch, sources   shown on the 'provenance' line;
                             dirty git or no sources -> WARN (not for converted files)
  nn_version at the root     duplicate of provenance.nn_args          WARN

Tests, in this order: load, attributes, provenance, weights NaN/Inf, code (same interface as the
current game?), nn_version (constructible by the current code?), sources (stored vs current
files), state_dict (layout of the net built from the CURRENT code with the STORED nn_args),
forward (dummy + real positions), [--complexity], [--onnx].

"Does this checkpoint still match my code?" is answered by three independent layers:
  code        board_size / action_size / num_players recorded at save time vs the current game
  state_dict  keys and shapes of the current net vs the stored weights (what torch would load)
  sources     stored copy of the network files vs the files on disk (same text / formatting only /
              changed). A changed file with an identical state_dict layout is a WARN: the weights
              load, but the function they compute may not be the one that was trained.

Rules
  - The checkpoint is only READ. The player count is never overridden: the current code
    (NUMBER_PLAYERS) must match the checkpoint, otherwise the run STOPS -- edit the constant by
    hand, delete the stale __pycache__ dirs (numba caches action_size()/observation_size()).
  - Net construction uses provenance.nn_args. Optional branches whose flag is not stored there
    show up as 'unexpected keys'.

--diff: unified diff stored sources -> current files, for the files that really changed.
--restore DIR: write the stored sources under DIR (same relative paths; absolute or '..' paths are
refused, existing files are never overwritten). Running an old net from them also needs the game
code of that time (rules, encoding): the git commit is printed for that (git worktree add).

--complexity: parameters (exact) and FLOPs of one forward pass (batch 1, torch's FlopCounterMode:
matmul/conv/bmm/attention only, FLOPs = 2 x MACs, so a lower bound).

--onnx FILE: nothing is exported without it. The net is exported like
GenericNNetWrapper.export_and_load_onnx and converted to each --onnx-opset in turn until one loads
in onnxruntime AND passes (1) a synthetic sign-bias test (signed boards vs their abs() twins:
integer floor-division bugs only hit negative values) and (2) a torch/ORT comparison on the forward
positions (WARN > 1e-4, FAIL > 1e-2). FILE is written only after the tests passed; an existing
FILE is overwritten with a warning.

--other-keys [main.py]: root keys other than the standard ones (Coach saves vars(args)), only when
different from the defaults of main.py (parsed with ast, never imported; the post-processing of
main() is mirrored for tempThreshold, maxlenOfQueue, stop_after_N_fail, load_model). Without
usable defaults, every key is shown. Put this option last.
"""

import argparse
import ast
import contextlib
import difflib
import gc
import io
import os
import re
import sys
import tempfile
import warnings
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import torch

ROOT = os.path.dirname(os.path.realpath(__file__))   # framework root: same as GenericNNetWrapper.ROOT

N_GAMES, MAX_PLIES, MAX_POS = 3, 400, 64         # random games used to get real positions
TOL_WARN, TOL_FAIL = 1e-4, 1e-2                   # torch/ONNX: max |d pi| over legal moves, max |d v|
GUARD_TOL, GUARD_N = 1e-4, 256                    # synthetic sign-bias test
OBSOLETE_RE = re.compile(r'unsupported|supports nn-version|not supported|unknown (nn[ -])?version', re.I)
STANDARD_KEYS = {'state_dict', 'provenance', 'sources', 'optim_state'}
DEFAULT_NN_ARGS = dict(lr=None, dropout=0., epochs=None, batch_size=None, learn_rate=None,
                       no_compression=False, q_weight=0.)
MISSING = object()
STATUSES = []
CTX = {}                                          # ckpt / drift report shared with --diff, --restore


class Stop(Exception):
	"""The run cannot continue without a manual action (player count, stale cache)."""


def say(name, status, detail=''):
	STATUSES.append(status)
	print(f'  {status:4s}  {name}' + (f': {detail}' if detail else ''), flush=True)


def short(e, n=160):
	lines = str(e).strip().splitlines()
	return f'{type(e).__name__}: {(lines[0] if lines else "")[:n]}'


def cut(s, n):
	s = str(s)
	return s if len(s) <= n else s[:n - 1] + '…'



# ----------------------------------------------------------------------------- loading

def is_old_format(path):
	"""Old files pickle the whole nn.Module under 'full_model'."""
	try:
		with zipfile.ZipFile(path) as z:
			return any(b'full_model' in z.read(n) for n in z.namelist() if n.endswith('data.pkl'))
	except Exception:
		return False


def load_checkpoint(path):
	"""The loaded dict, or None when unreadable (the FAIL line is already printed)."""
	try:
		return torch.load(path, map_location='cpu', weights_only=True)
	except (EOFError, zipfile.BadZipFile) as e:
		say('load', 'FAIL', f'unreadable or truncated file ({short(e)})')
	except RuntimeError as e:
		if re.search(r'PytorchStreamReader|central directory|unexpected EOF|invalid magic', str(e), re.I):
			say('load', 'FAIL', f'unreadable or truncated file ({short(e)})')
		else:
			say('load', 'FAIL', short(e))
	except Exception as e:                                       # pickle.UnpicklingError: not plain types
		if is_old_format(path):
			say('load', 'FAIL', 'old-format checkpoint (pickled full_model): convert it with convert_checkpoint.py')
		else:
			say('load', 'FAIL', f'not loadable with weights_only=True ({short(e)})')
	return None


# ----------------------------------------------------------------------------- attributes

def int_state(v, lo):
	if v is MISSING:
		return 'absent'
	if v is None:
		return 'none'
	if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
		return 'bad_type'
	return 'out_of_range' if v < lo else 'ok'


def shape_state(v):
	if v is MISSING:
		return 'absent'
	if v is None:
		return 'none'
	if not isinstance(v, (tuple, list, np.ndarray)) or len(v) == 0:
		return 'bad_type'
	if any(int_state(x, 1) != 'ok' for x in v):
		return 'bad_type' if any(int_state(x, 1) == 'bad_type' for x in v) else 'out_of_range'
	return 'ok'


def check_attributes(ckpt):
	"""Standard keys. Returns the resolved values (None where unusable)."""
	n0 = len(STATUSES)
	A = dict(version=None, actions=None, players=None, shape=None, nn_args=None, sd_ok=False,
	         prov={}, sources=None, converted=False, game=None)

	sd = ckpt.get('state_dict', MISSING)
	if sd is MISSING:
		say('state_dict', 'FAIL', 'absent')
	elif not (isinstance(sd, dict) and sd and all(torch.is_tensor(v) for v in sd.values())):
		say('state_dict', 'FAIL', 'empty, or not a dict of tensors')
	else:
		A['sd_ok'] = True

	prov = ckpt.get('provenance', MISSING)
	if prov is MISSING or not isinstance(prov, dict):
		say('provenance', 'FAIL', 'absent' if prov is MISSING else f'unexpected type {type(prov).__name__}')
		return A
	A['prov'], A['converted'] = prov, 'converted' in prov

	def field(name, lo):
		v, st = prov.get(name, MISSING), int_state(prov.get(name, MISSING), lo)
		if st == 'ok':
			return int(v)
		say(f'provenance.{name}', 'FAIL', st if v is MISSING else f'{st} ({v!r})')

	A['actions'], A['players'] = field('action_size', 1), field('num_players', 2)
	bs = prov.get('board_size', MISSING)
	if shape_state(bs) == 'ok':
		A['shape'] = tuple(int(x) for x in bs)
	else:
		say('provenance.board_size', 'FAIL', f'{shape_state(bs)} ({bs!r})' if bs is not MISSING else 'absent')
	na = prov.get('nn_args', MISSING)
	if isinstance(na, dict) and int_state(na.get('nn_version', MISSING), 1) == 'ok':
		A['nn_args'], A['version'] = dict(na), int(na['nn_version'])
	else:
		say('provenance.nn_args', 'FAIL', 'absent, or without an int nn_version')

	g = prov.get('game')
	if isinstance(g, str) and g:
		A['game'] = g
	elif g is not None:
		say('provenance.game', 'WARN', f'unexpected value {g!r}')
	if 'nn_version' in ckpt:                                     # the wrapper filters it out of vars(args)
		say('nn_version', 'WARN', f'duplicate root key ({ckpt["nn_version"]!r}); the only source is provenance.nn_args')
	src = ckpt.get('sources', MISSING)
	if src is not MISSING:
		if isinstance(src, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in src.items()):
			A['sources'] = src
		else:
			say('sources', 'FAIL', 'not a {path: text} dict')

	if sum(s in ('FAIL', 'WARN') for s in STATUSES[n0:]) == 0:
		saved_at = prov.get('saved_at')
		say('attributes', 'OK', f'V{A["version"]}, P={A["players"]}, in={list(A["shape"])}, actions={A["actions"]}, '
		    f'saved_at {saved_at if isinstance(saved_at, str) and saved_at else "-"}, game {A["game"] or "-"}')
	return A


def check_provenance(A):
	"""One line: git state, torch, number of stored source files, command line."""
	p, src = A['prov'], A['sources'] or {}
	git = p.get('git') if isinstance(p.get('git'), dict) else {}
	commit, dirty = git.get('commit'), git.get('dirty')
	parts = [f'git {commit[:9] if isinstance(commit, str) else "-"}' + (' (uncommitted changes)' if dirty else '')]
	parts.append(f'torch {p.get("torch") or "-"}')
	parts.append(f'{len(src)} source file(s)')
	cmd = p.get('command_line')
	parts.append(f'cmd {cut(cmd, 110)}' if isinstance(cmd, str) and cmd else 'cmd -')
	problems = []
	if not A['converted']:
		if dirty:
			problems.append('git commit alone does not reproduce the code')
		if not src:
			problems.append('no sources recorded')
	if A['converted']:
		parts.append(f'converted {str(p["converted"])[:25]}')
	say('provenance', 'WARN' if problems else 'OK', ', '.join(parts) + (f'  <- {"; ".join(problems)}' if problems else ''))


def check_vs_code(A, game):
	"""Compare with the CURRENT game code. A player-count mismatch stops the run."""
	p, act = int(game.num_players), int(game.getActionSize())
	shape = tuple(int(x) for x in np.atleast_1d(game.getBoardSize()))
	if A['players'] is not None and A['players'] != p:
		raise Stop(f'checkpoint has num_players={A["players"]} (action_size={A["actions"]}) but the code has '
		           f'NUMBER_PLAYERS={p} (action_size={act}). Edit the constant by hand, delete the stale '
		           f'__pycache__ dirs, rerun.')
	diffs = []
	if A['actions'] is not None and A['actions'] != act:
		diffs.append(f'action_size {A["actions"]} vs code {act}')
	if A['shape'] is not None and tuple(A['shape']) != shape:
		diffs.append(f'input {A["shape"]} vs code {shape}')
	if diffs:
		say('code', 'FAIL', 'stale checkpoint (rules/encoding changed): ' + '; '.join(diffs))
		return False
	say('code', 'OK', f'P={p}, input={shape}, actions={act}')
	return True


def check_environment(game):
	"""The code must agree with itself: a stale numba cache freezes observation_size()/action_size()."""
	board = game.getInitBoard()
	size = tuple(int(x) for x in np.atleast_1d(game.getBoardSize()))
	if tuple(board.shape) != size:
		return f'getBoardSize()={size} but getInitBoard().shape={tuple(board.shape)}: stale numba cache? delete the __pycache__ dirs'
	n = len(game.getValidMoves(board, 0))
	if n != int(game.getActionSize()):
		return f'getActionSize()={int(game.getActionSize())} but valid_moves has {n} entries: stale numba cache?'
	return None


# ----------------------------------------------------------------------------- weights

def check_weights(sd):
	bad, norms = [], {}
	for k, t in sd.items():
		if not (t.is_floating_point() or t.is_complex()):
			continue
		fin = torch.isfinite(t)
		if not bool(fin.all()):
			bad.append((k, int((~fin).sum()), t.numel()))
			t = t[fin]
		norms[k] = float(t.float().norm()) if t.numel() else 0.0
	if not bad:
		return say('weights', 'OK', f'{len(sd)} tensors finite')
	lo, hi = min(norms, key=norms.get), max(norms, key=norms.get)
	say('weights', 'FAIL', f'NaN/Inf in {len(bad)} tensors (first: {bad[0][0]}, {bad[0][1]}/{bad[0][2]}); '
	    f'layer norms (finite part) min {norms[lo]:.3g} [{lo}], max {norms[hi]:.3g} [{hi}]')


# ----------------------------------------------------------------------------- state_dict / sources

def check_state_dict(module, sd, drifted):
	net_sd = module.state_dict()
	missing, unexpected = sorted(set(net_sd) - set(sd)), sorted(set(sd) - set(net_sd))
	shape = sorted(k for k in net_sd if k in sd and tuple(net_sd[k].shape) != tuple(sd[k].shape))
	prefixes = tuple(getattr(module, 'additive_param_prefixes', ()))      # zero-gated optional branches
	additive = [k for k in missing if prefixes and k.startswith(prefixes)]
	hard = [k for k in missing if k not in additive]
	problems = []
	if hard:
		zero = [k for k in hard if net_sd[k].is_floating_point() and not bool(net_sd[k].any())]
		problems.append(f'{len(hard)} missing keys ({", ".join(hard[:8])}{", ..." if len(hard) > 8 else ""}); '
		                f'{len(zero)} zero in the freshly built net (zero-init additive branch? '
		                f'declare it in additive_param_prefixes)')
	if unexpected:
		problems.append(f'{len(unexpected)} unexpected keys (e.g. {unexpected[0]}; optional branch not stored in nn_args?)')
	if shape:
		problems.append(f'{len(shape)} shape mismatches (e.g. {shape[0]}: ckpt {tuple(sd[shape[0]].shape)} '
		                f'vs net {tuple(net_sd[shape[0]].shape)})')
	if problems:
		say('state_dict', 'FAIL', '; '.join(problems) + (' -- sources changed since save (see above, --diff)' if drifted else ''))
		return False
	if additive:
		say('state_dict', 'WARN', f'{len(additive)} additive zero-gated tensors absent (e.g. {additive[0]})')
	else:
		say('state_dict', 'OK', f'{len(net_sd)} keys and shapes match')
	module.load_state_dict(sd, strict=False)
	return True


def safe_rel(rel):
	"""Stored source paths are relative, '/'-separated, inside the tree: never trust them blindly."""
	parts = rel.split('/')
	return bool(rel) and not rel.startswith(('/', '\\')) and '\\' not in rel and ':' not in rel \
	       and all(p not in ('', '.', '..') for p in parts)


def ast_equal(a, b):
	try:
		return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
	except SyntaxError:
		return False


def compare_sources(stored, now_sources=None):
	"""Stored copy vs the files on disk (plus, when known, the files the current code imports)."""
	r = dict(same=[], format_only=[], changed={}, missing=[], unsafe=[], new=[])
	for rel, old in sorted(stored.items()):
		if not safe_rel(rel):
			r['unsafe'].append(rel)
			continue
		try:
			with open(os.path.join(ROOT, *rel.split('/')), encoding='utf-8') as f:
				new = f.read()
		except OSError:
			r['missing'].append(rel)
			continue
		if new == old:
			r['same'].append(rel)
		elif ast_equal(old, new):
			r['format_only'].append(rel)
		else:
			d = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm='', n=0)
			     if l[:1] in ('+', '-') and l[:3] not in ('+++', '---')]
			r['changed'][rel] = (sum(l[0] == '+' for l in d), sum(l[0] == '-' for l in d))
	if now_sources:
		r['new'] = sorted(set(now_sources) - set(stored))
	return r


def check_sources(A, now_sources):
	"""Returns True when a source file changed or vanished (explains a state_dict mismatch)."""
	if not A['sources']:
		say('sources', 'SKIP', 'no stored copy')
		return False
	r = CTX['drift'] = compare_sources(A['sources'], now_sources)
	bad = []
	if r['changed']:
		bad.append(f'{len(r["changed"])} changed (' + ', '.join(f'{k} +{a}/-{b}' for k, (a, b) in r['changed'].items()) + ')')
	if r['missing']:
		bad.append('missing now: ' + ', '.join(r['missing']))
	if r['unsafe']:
		bad.append('unsafe stored paths ignored: ' + ', '.join(r['unsafe']))
	if r['new']:
		bad.append('imported by the current code but not stored: ' + ', '.join(r['new']))
	if bad:
		say('sources', 'WARN', '; '.join(bad) + '  (--diff for details)')
	else:
		note = f'; {len(r["format_only"])} differ in formatting/comments only ({", ".join(r["format_only"])})' if r['format_only'] else ''
		say('sources', 'OK', f'{len(A["sources"])} file(s) identical to disk{note}')
	return bool(r['changed'] or r['missing'])


def show_diff(ckpt):
	src = ckpt.get('sources') if isinstance(ckpt.get('sources'), dict) else None
	if not src:
		return say('diff', 'SKIP', 'no stored sources')
	r = CTX.get('drift') or compare_sources(src)
	if not (r['changed'] or r['missing']):
		return say('diff', 'OK', 'no real difference between the stored sources and the files on disk')
	for rel in r['changed']:
		with open(os.path.join(ROOT, *rel.split('/')), encoding='utf-8') as f:
			lines = list(difflib.unified_diff(src[rel].splitlines(), f.read().splitlines(),
			                                  f'stored/{rel}', f'current/{rel}', lineterm=''))
		print('\n'.join(lines[:400]) + (f'\n... ({len(lines) - 400} more lines)' if len(lines) > 400 else ''))
	say('diff', 'WARN', f'{len(r["changed"])} file(s) changed, {len(r["missing"])} missing')


def restore_sources(ckpt, dest):
	src = ckpt.get('sources') if isinstance(ckpt.get('sources'), dict) else None
	if not src:
		return say('restore', 'FAIL', 'no stored sources in this checkpoint')
	dest, done, exists, refused = os.path.abspath(dest), [], [], []
	for rel, text in sorted(src.items()):
		if not safe_rel(rel):
			refused.append(rel)
			continue
		out = os.path.join(dest, *rel.split('/'))
		if os.path.exists(out):
			exists.append(rel)
			continue
		os.makedirs(os.path.dirname(out), exist_ok=True)
		with open(out, 'w', encoding='utf-8', newline='\n') as f:
			f.write(text)
		done.append(rel)
	git = (ckpt.get('provenance') or {}).get('git') or {}
	msg = f'{len(done)} file(s) written under {dest}'
	if exists:
		msg += f'; NOT overwritten (already exist): {", ".join(exists)}'
	if refused:
		msg += f'; refused unsafe paths: {", ".join(refused)}'
	if git.get('commit'):
		msg += f'. Game code of that time: git worktree add <dir> {git["commit"]}' + (' (dirty at save time)' if git.get('dirty') else '')
	say('restore', 'FAIL' if (exists or refused) else 'OK', msg)


# ----------------------------------------------------------------------------- forward

def collect_positions(game, rng):
	"""Real positions from random legal games, as the net sees them (observation if hidden info)."""
	hidden = hasattr(game, 'getObservation')
	boards, valids = [], []
	for _ in range(N_GAMES):
		board, player = np.copy(game.getInitBoard()), 0
		for _ply in range(MAX_PLIES):
			if np.any(game.getGameEnded(board, player)):
				break
			cb = np.copy(game.getCanonicalForm(board, player))
			valid = np.asarray(game.getValidMoves(cb, 0), dtype=np.bool_).copy()
			if not valid.any():
				break
			boards.append((np.copy(game.getObservation(cb, 0)) if hidden else cb).astype(np.float32))
			valids.append(valid)
			board, player = game.getNextState(board, player, int(rng.choice(np.flatnonzero(valid))), random_seed=0)
			board, player = np.copy(board), int(player)
	if not boards:
		raise RuntimeError('no position generated')
	idx = np.unique(np.linspace(0, len(boards) - 1, min(MAX_POS, len(boards))).astype(int))
	return np.stack([boards[i] for i in idx]), np.stack([valids[i] for i in idx])


def run_forward(module, boards, valids, P, act):
	"""Returns (fails, warns, log_policy, value)."""
	module.eval()
	with torch.no_grad():
		lp, v = module(torch.from_numpy(boards), torch.from_numpy(valids))
	n = len(boards)
	if tuple(lp.shape) != (n, act):
		return [f'policy shape {tuple(lp.shape)}, expected {(n, act)}'], [], lp, v
	if v.shape[0] != n or v.numel() != n * P:
		return [f'value shape {tuple(v.shape)}, expected ({n}, {P})'], [], lp, v
	fails, warns = [], []
	if not torch.isfinite(lp).all():
		fails.append('non-finite policy')
	if not torch.isfinite(v).all():
		fails.append('non-finite value')
	if fails:
		return fails, warns, lp, v
	p, vm = lp.exp().double(), torch.from_numpy(valids)
	rows = vm.any(1)
	if bool(rows.any()):
		err = ((p * vm).sum(1)[rows] - 1).abs().max()
		if err > 1e-4:
			fails.append(f'policy not normalised on legal moves (max err {float(err):.1e})')
		leak = (p * ~vm).sum(1)[rows].max()
		if leak > 1e-4:
			fails.append(f'probability mass on illegal moves ({float(leak):.1e})')
	if float(v.abs().max()) > 1 + 1e-5:
		warns.append('value outside [-1, 1]')
	return fails, warns, lp, v


def forward_checks(module, game, rng):
	"""Dummy + real positions. Returns (boards, valids, log_policy) of the last good pass, or None."""
	P, act = int(game.num_players), int(game.getActionSize())
	shape = tuple(int(x) for x in np.atleast_1d(game.getBoardSize()))
	used = None

	def one(label, boards, valids, ok_detail):
		nonlocal used
		try:
			fails, warns, lp, _ = run_forward(module, boards, valids, P, act)
		except Exception as e:
			return say(label, 'FAIL', f'forward failed: {short(e)}')
		say(label, 'FAIL' if fails else 'WARN' if warns else 'OK', '; '.join(fails or warns) or ok_detail)
		if not fails:
			used = (boards, valids, lp)

	one('forward(dummy)', np.zeros((2,) + shape, np.float32), np.ones((2, act), np.bool_),
	    f'policy ({act}), value ({P}), normalised')
	try:
		boards, valids = collect_positions(game, rng)
	except Exception as e:
		say('forward(real)', 'WARN', f'skipped: {short(e)}')
		return used
	one(f'forward(real, {len(boards)} pos)', boards, valids, 'shapes, finiteness, normalisation ok')
	return used

# ----------------------------------------------------------------------------- complexity / onnx

def fmt_count(n, unit=''):
	return f'{n / 1e9:.2f} G{unit}' if n >= 1e9 else f'{n / 1e6:.2f} M{unit}' if n >= 1e6 else f'{n / 1e3:.1f} k{unit}'


def complexity(module, game):
	from torch.utils.flop_counter import FlopCounterMode
	params = sum(p.numel() for p in module.parameters())
	msg = f'{fmt_count(params)} params ({params * 4 / 1e6:.1f} MB fp32), {sum(b.numel() for b in module.buffers())} buffer values'
	shape = tuple(int(x) for x in np.atleast_1d(game.getBoardSize()))
	board, valid = torch.zeros((1,) + shape), torch.ones((1, int(game.getActionSize())), dtype=torch.bool)
	try:
		module.eval()
		with torch.no_grad(), FlopCounterMode(display=False) as fc:
			module(board, valid)
		say('complexity', 'OK', f'{msg}; ~{fmt_count(fc.get_total_flops(), "FLOPs")}/forward (batch 1, matmul+conv only)')
	except Exception as e:
		say('complexity', 'OK', f'{msg}; FLOPs not measurable ({short(e, 80)})')



def sign_bias_test(net, sess, board_shape, act, seed=0):
	"""Synthetic test (it used to run inside GenericNNetWrapper at every ONNX export).
	Boards come in PAIRS, a signed board and its abs() twin, at amplitudes 8 and 128, batch 8 and 1.
	Integer ops that differ between torch and ONNX (floor vs truncating division) only touch
	negative values: they show as signed >> abs. Uniform int8 noise is out of distribution, so an
	error over tolerance on BOTH twins is float32 conditioning and is not a failure.
	Returns (sign_biased, detail)."""
	rng = np.random.default_rng(seed)
	base = rng.integers(-128, 128, size=(GUARD_N,) + tuple(board_shape)).astype(np.float32)
	valids = rng.random((GUARD_N, act)) > 0.5
	valids[:, 0] = True                                           # no all-illegal row
	net.eval()

	def errors(boards, bs):
		with torch.no_grad():
			pi_t, v_t = net(torch.from_numpy(boards), torch.from_numpy(valids))
		pi_t, v_t = pi_t.numpy(), v_t.numpy()
		pi_o, v_o = np.empty_like(pi_t), np.empty_like(v_t)
		for s in range(0, GUARD_N, bs):
			e = min(s + bs, GUARD_N)
			out = sess.run(None, {'board': boards[s:e], 'valid_actions': valids[s:e]})
			pi_o[s:e], v_o[s:e] = out[0], out[1]
		dpi = np.where(valids, np.abs(np.exp(pi_t) - np.exp(pi_o)), 0.).max(axis=1)
		dv = np.abs(v_t - v_o).reshape(GUARD_N, -1).max(axis=1)
		return dpi, dv

	biased, worst_s, worst_a = [], 0., 0.
	for amp in (8, 128):
		signed = np.clip(np.round(base * amp / 128.), -amp, amp - 1).astype(np.float32)
		for bs in (8, 1):
			dpi_s, dv_s = errors(signed, bs)
			dpi_a, dv_a = errors(np.abs(signed), bs)
			err_s, err_a = np.maximum(dpi_s, dv_s) / GUARD_TOL, np.maximum(dpi_a, dv_a) / GUARD_TOL
			worst_s, worst_a = max(worst_s, float(err_s.max())), max(worst_a, float(err_a.max()))
			if err_s.max() > 1 and np.median(err_s) > 10 * np.median(err_a) + 1e-3:
				biased.append(f'amp {amp} batch {bs}')
	detail = f'sign-bias test: max err signed {worst_s * GUARD_TOL:.1e} / abs twin {worst_a * GUARD_TOL:.1e}'
	return bool(biased), detail + (f' (SIGN-BIASED at {", ".join(biased)})' if biased else '')


def export_onnx(wrapper, used, opsets, out_path):
	"""First opset that loads in ORT and passes both parity tests wins; its file is then written."""
	try:
		import onnx
		import onnxruntime as ort
	except ImportError as e:
		return say('onnx', 'FAIL', f'{short(e)}')
	boards, valids, lp_t = used
	pi_t = np.exp(lp_t.numpy())
	with torch.no_grad():
		v_t = wrapper.nnet(torch.from_numpy(boards), torch.from_numpy(valids))[1].numpy()
	net = wrapper.nnet.to('cpu').eval()
	dummy = (torch.randn(wrapper.board_size).unsqueeze(0), (torch.randn(wrapper.action_size) > 0.5).unsqueeze(0))
	tried, kept = [], None
	try:
		with tempfile.TemporaryDirectory() as tmp:
			f = os.path.join(tmp, 'export.onnx')
			with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
				warnings.simplefilter('ignore')                  # exporter logs are noise here
				torch.onnx.export(net, dummy, f, input_names=['board', 'valid_actions'], output_names=['pi', 'v'],
								  dynamic_axes={k: {0: 'batch_size'} for k in ('board', 'valid_actions', 'pi', 'v')})
			for op in opsets:
				model = onnx.load(f)
				label = f'opset {op}'
				try:
					if op != 'native':
						model = onnx.version_converter.convert_version(model, int(op))
					else:
						label = 'opset ' + str(next((o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')), '?'))
					blob = model.SerializeToString()
					opts = ort.SessionOptions()
					opts.intra_op_num_threads = 1
					sess = ort.InferenceSession(blob, sess_options=opts, providers=['CPUExecutionProvider'])
				except Exception as e:
					tried.append(f'{label}: {short(e, 80)}')
					continue
				biased, gdetail = sign_bias_test(net, sess, wrapper.board_size, wrapper.action_size)
				dpi, dv = [], []
				for i in range(0, len(boards), 64):
					out = sess.run(None, {'board': boards[i:i + 64], 'valid_actions': valids[i:i + 64]})
					dpi.append(np.where(valids[i:i + 64], np.abs(pi_t[i:i + 64] - np.exp(out[0])), 0.).max(1))
					dv.append(np.abs(v_t[i:i + 64].reshape(len(out[1]), -1) - out[1].reshape(len(out[1]), -1)).max(1))
				dpi, dv = float(np.concatenate(dpi).max()), float(np.concatenate(dv).max())
				detail = f'{label}, max|dpi| {dpi:.1e}, max|dv| {dv:.1e} on {len(boards)} positions; {gdetail}'
				if biased:
					tried.append(f'{label}: {gdetail}')
				elif dpi > TOL_FAIL or dv > TOL_FAIL:
					tried.append(f'{label}: {detail}')
				else:
					kept = (blob, detail, dpi > TOL_WARN or dv > TOL_WARN)
					break
	except Exception as e:
		return say('onnx', 'FAIL', f'export failed: {short(e)}')
	if kept is None:
		return say('onnx', 'FAIL', 'no opset passed -- ' + ' | '.join(tried))
	blob, detail, suspect = kept
	if os.path.exists(out_path):
		print(f'  warning: overwriting existing {out_path}', file=sys.stderr)
	with open(out_path, 'wb') as f:                              # written only after the tests passed
		f.write(blob)
	say('onnx', 'WARN' if (suspect or tried) else 'OK', f'written {out_path}; parity tests passed; {detail}'
		+ (f'; earlier failures: {" | ".join(tried)}' if tried else ''))


# ----------------------------------------------------------------------------- other keys

def default_values(main_py):
	"""Defaults of main.py's argparse options by static parsing (nothing is imported or run), then
	the post-processing main() applies after parse_args (tempThreshold, maxlenOfQueue, ...)."""
	out = {}
	try:
		tree = ast.parse(open(main_py, encoding='utf-8').read())
	except Exception:
		return out
	for node in ast.walk(tree):
		if not (isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'add_argument'):
			continue
		names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
		kw = {k.arg: k.value for k in node.keywords}
		dest = kw['dest'].value if isinstance(kw.get('dest'), ast.Constant) else None
		if not dest and names:
			dest = ([n for n in names if n.startswith('--')] or names)[0].lstrip('-').replace('-', '_')
		action = kw['action'].value if isinstance(kw.get('action'), ast.Constant) else 'store'
		try:
			if dest and 'default' in kw:
				out[dest] = ast.literal_eval(kw['default'])
			elif dest and action in ('store_true', 'store_false'):
				out[dest] = action == 'store_false'
			elif dest and 'required' not in kw:
				out[dest] = None
		except Exception:
			pass
	try:                                                     # mirror of main.py post-processing
		t = out.get('temperature')
		if t is not None and len(t) == 4:
			out['tempThreshold'] = int(float(out['tempThreshold']) if out.get('tempThreshold') is not None else t[3])
		if 'numItersHistory' in out:
			h = out['numItersHistory']
			out['maxlenOfQueue'] = int(2.5e6 / ((2 if out.get('no_compression') else 0.5) * h))
			if out.get('stop_after_N_fail') is not None and out['stop_after_N_fail'] < 0:
				out['stop_after_N_fail'] = -out['stop_after_N_fail'] * h
		if 'load_folder_file' in out:
			out['load_model'] = out['load_folder_file'] is not None
	except Exception:
		pass
	return out


def print_other_keys(extra, main_py):
	defaults = default_values(main_py) if os.path.isfile(main_py) else {}
	print(f'\nOther keys differing from the defaults of {main_py} ({len(defaults)} defaults parsed):' if defaults
	      else f'\nOther keys (no usable defaults in {main_py!r}: all shown):')
	shown = 0
	for k, v in sorted(extra.items()):
		try:
			if k in defaults and bool(v == defaults[k]):
				continue
		except Exception:
			pass
		shown += 1
		print(f'  {k} = {cut(repr(v), 120)}' + ('   (no default in main.py)' if defaults and k not in defaults else ''))
	if not shown:
		print('  (none)')



# ----------------------------------------------------------------------------- run

def verify(a, extra):
	path = os.path.abspath(a.checkpoint)
	if not os.path.isfile(path):
		return say('file', 'FAIL', 'file not found')
	ckpt = load_checkpoint(path)
	if ckpt is None:
		return
	if not isinstance(ckpt, dict) or 'state_dict' not in ckpt:
		return say('file', 'FAIL', f'not a framework checkpoint ({type(ckpt).__name__})')
	say('load', 'OK', 'torch.load(weights_only=True) ok')
	CTX['ckpt'] = ckpt
	extra.update({k: v for k, v in ckpt.items() if k not in STANDARD_KEYS})

	A = check_attributes(ckpt)
	if A['prov']:
		check_provenance(A)
	sd = ckpt.get('state_dict')
	if A['sd_ok']:
		check_weights(sd)

	game_name = a.game or A['game']
	if not game_name:
		return say('code', 'FAIL', 'no game name in the checkpoint (converted file?): pass it, e.g. verify_checkpoints.py smallworld f.pt')
	if a.game and A['game'] and a.game != A['game']:
		say('game', 'WARN', f'checkpoint was saved by "{A["game"]}", you ask "{a.game}"')
	try:
		from GameSwitcher import import_game
		Game, NNet, _, _ = import_game(game_name)
		game = Game()
	except Exception as e:
		return say('code', 'FAIL', f'game code not usable: {short(e, 220)}')
	env = check_environment(game)
	if env:
		raise Stop(env)
	if not check_vs_code(A, game) or A['version'] is None:
		return A['version'] is None and say('network', 'FAIL', 'no usable provenance.nn_args')

	nn_args = {**DEFAULT_NN_ARGS, **A['nn_args']}                # what the net was trained with
	try:
		wrapper = NNet(game, nn_args)
	except Exception as e:
		if OBSOLETE_RE.search(str(e)):
			return say('nn_version', 'WARN', f'obsolete version V{A["version"]}: {short(e)}')
		return say('network', 'FAIL', f'construction failed: {short(e)}')
	module = wrapper.nnet
	if sum(p.numel() for p in module.parameters()) == 0:
		return say('nn_version', 'WARN', f'obsolete version V{A["version"]}: current code builds an empty network')
	say('nn_version', 'OK', f'V{A["version"]} supported by current code')

	drifted = check_sources(A, wrapper.sources)
	if not A['sd_ok'] or not check_state_dict(module, sd, drifted):
		return
	used = forward_checks(module, game, np.random.default_rng(0))
	if a.complexity:
		complexity(module, game)
	if a.onnx:
		if used is None:
			say('onnx', 'SKIP', 'forward pass failed')
		else:
			export_onnx(wrapper, used, [o.strip() for o in a.onnx_opset.split(',') if o.strip()], a.onnx)


def main():
	ap = argparse.ArgumentParser(description='Read-only verifier of one checkpoint file (.pt).')
	ap.add_argument('game', nargs='?', default=None, help='game name (default: provenance.game of the file)')
	ap.add_argument('checkpoint', help='explicit .pt file')
	ap.add_argument('--complexity', action='store_true', help='parameter count and FLOPs of one forward pass')
	ap.add_argument('--onnx', default=None, metavar='FILE',
	                help='export to this ONNX file after torch/ORT parity tests (nothing exported if absent)')
	ap.add_argument('--onnx-opset', default='21,20,19,18,17,native', metavar='LIST',
	                help='opsets tried in order until one loads and passes parity (native = exporter default)')
	ap.add_argument('--diff', action='store_true', help='unified diff stored sources -> current files')
	ap.add_argument('--restore', default=None, metavar='DIR', help='write the stored sources under DIR (never overwrites)')
	ap.add_argument('--other-keys', nargs='?', const='main.py', default=None, metavar='MAIN_PY',
	                help='non-standard file keys that differ from the defaults of main.py (put this option last)')
	a = ap.parse_args()
	if os.path.isdir(a.checkpoint):
		ap.error('give an explicit .pt file, not a directory')
	if a.onnx and a.onnx.lower().endswith(('.pt', '.pth', '.tar')):
		ap.error('--onnx must not look like a checkpoint file')

	extra = {}
	try:
		verify(a, extra)
	except Stop as e:
		say('code', 'FAIL', str(e))
		print('STOP: fix the code/environment, then rerun', file=sys.stderr)
	except Exception as e:                                       # report it, do not crash with a traceback
		say('internal', 'FAIL', f'unexpected error: {short(e)}')
	gc.collect()

	if 'ckpt' in CTX:                                            # need the file only, not the game code
		try:
			if a.diff:
				show_diff(CTX['ckpt'])
			if a.restore:
				restore_sources(CTX['ckpt'], a.restore)
		except Exception as e:
			say('internal', 'FAIL', f'--diff/--restore failed: {short(e)}')
	if a.other_keys is not None:
		print_other_keys(extra, a.other_keys)
	elif extra:
		print(f'\n{len(extra)} non-standard key(s) not shown (--other-keys [main.py] to display them)')
	worst = 'FAIL' if 'FAIL' in STATUSES else 'WARN' if 'WARN' in STATUSES else 'OK'
	print(f'\nSTATUS  {worst}')
	sys.exit(2 if worst == 'FAIL' else 0)


if __name__ == '__main__':
	main()
