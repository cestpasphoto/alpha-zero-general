#!/usr/bin/env python3
"""
convert_checkpoint.py -- old-format or previous-layout checkpoint -> current layout.

  python convert_checkpoint.py OLD.pt -o NEW.pt [--force]
  python convert_checkpoint.py a.pt b.pt c.pt -o OUT_DIR/ [--force]
  options: --game NAME  --python V --torch V --numpy V --numba V --onnx V --onnxruntime V   (see below)

Run from the framework root (next to GameSwitcher.py). Writing over the input file (-o same path)
is allowed with --force; any existing output needs --force. No game rule is executed: classes that
cannot be imported are read through stubs, only attributes and tensors are used.

Inputs
  old format         pickled full_model (+ state_dict)
  previous layout    provenance holding board_size / action_size / num_players / nn_args / game / git
                     (+ sources as {path: text}), the Coach's args as root keys
  current layout     refused ("already in the new format")

Output layout (see GenericNNetWrapper.py)
  state_dict      copied unchanged
  game            from the file, else the Coach's args, else settings.txt, else --game, else None
  board_size, action_size, num_players   from the file / the pickled model (a missing one is an error)
  nn_args         in this order: the model's own args (old format) or the file's nn_args when it was
                  really saved by a wrapper; else REBUILT like main.py does from the Coach's args
                  (nn_opt included); else the file's nn_args if it came from a conversion; else
                  defaults. nn_args['nn_version'] = the net's own version.
  args            the Coach's args found in the file (root keys), completed from settings.txt for the
                  keys the file lacks; without nn_version
  provenance      saved_at, command_line (None when unknown, never invented), python/torch/numpy/numba/onnx/onnxruntime
                  (the file's own when recorded, else the values below), converted (kept, or written
                  for an old-format file)
  sources         previous layout: its own stored files. Otherwise, from the run directory (see below).
                  Else absent: the code of that time is unknown, never filled with today's files.

Run directory: the folder of the input file, when it holds a settings.txt (main.py writes it at launch
next to a flat copy of the code, `cp *py <game>/*py`). settings.txt (parsed with ast, never evaluated)
and the *.py files are used only if the run was not restarted there AFTER the file was saved: when
settings.<epoch> backups exist, the file's saved_at must be later than settings.txt. A flat file is
stored as <name> if it exists in the framework root, <game>/<name> if it exists in the game package,
otherwise skipped (its place in the tree is unknown).

Versions of the environment that trained the checkpoints: edit ENV_VERSIONS below, or pass
--python/--torch/--numpy/--numba/--onnx/--onnxruntime. They fill only what the file does not record.

Each output is re-read with torch.load(weights_only=True) and compared (weights, nn_version, sources)
before it replaces the temporary file. Exit code 0 if every file was converted, 2 otherwise.
"""

import argparse
import ast
import glob
import json
import os
import pickle
import re
import sys
import types
import zipfile
import zlib
from datetime import datetime

import numpy as np
import torch

ROOT = os.path.dirname(os.path.realpath(__file__))

# Versions in use when the checkpoints were trained (None = unknown). Overridden by the options.
ENV_VERSIONS = dict(python='3.13.1', torch='2.7.0', numpy='2.2.5', numba='0.61.2', onnx='1.18.0', onnxruntime='1.22.0')

REMOVED_OLD = {'full_model', 'nn_version', 'onnx_parity_skipped', 'state_dict', 'provenance', 'optim_state'}
STANDARD_PREVIOUS = {'state_dict', 'provenance', 'sources', 'optim_state'}
DEFAULT_NN_ARGS = dict(lr=None, dropout=0., epochs=None, batch_size=None, learn_rate=None,
                       no_compression=False, q_weight=0.)
NN_ARG_KEYS = ('learn_rate', 'dropout', 'epochs', 'batch_size', 'nn_version', 'no_compression', 'q_weight')


def now_iso():
	return datetime.now().astimezone().isoformat(timespec='seconds')


def shape_list(x):
	return [int(d) for d in (x if isinstance(x, (list, tuple)) else [x])]


class _Stub:
	"""Stands in for a class that cannot be imported."""
	def __setstate__(self, state):
		self.__dict__.update(state if isinstance(state, dict) else {'_state': state})


class _StubUnpickler(pickle.Unpickler):
	def find_class(self, module, name):
		try:
			return super().find_class(module, name)
		except Exception:                                        # missing module/constant, error in game code
			return type(name, (_Stub,), {'_stub_origin': f'{module}.{name}'})


def read_old(path):
	stub_module = types.ModuleType('stub_pickle')
	stub_module.__dict__.update({k: v for k, v in vars(pickle).items() if not k.startswith('__')})
	stub_module.Unpickler = _StubUnpickler
	return torch.load(path, map_location='cpu', weights_only=False, pickle_module=stub_module)


def plain(x):
	"""Builtin types and tensors only (what weights_only=True accepts). Returns (value, n_repr)."""
	if torch.is_tensor(x) or x is None or isinstance(x, (bool, int, float, str)):
		return x, 0
	if isinstance(x, dict):
		out, n = {}, 0
		for k, v in x.items():
			out[k if isinstance(k, (str, int)) else str(k)], m = plain(v)
			n += m
		return out, n
	if isinstance(x, (list, tuple)):
		pairs = [plain(v) for v in x]
		return [p[0] for p in pairs], sum(p[1] for p in pairs)
	if isinstance(x, np.generic):
		return x.item(), 0
	if isinstance(x, np.ndarray):
		return x.tolist(), 0
	return repr(x), 1


def same_tensor(a, b):
	"""Bit-identical, NaN included (torch.equal says NaN != NaN)."""
	if a.shape != b.shape or a.dtype != b.dtype:
		return False
	if a.is_floating_point() or a.is_complex():
		return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())
	return torch.equal(a, b)


def is_int(v, lo):
	return isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v >= lo


def pack_sources(files):
	"""{path: text} -> uint8 tensor holding zlib(JSON) (same as GenericNNetWrapper.pack_sources)."""
	blob = zlib.compress(json.dumps(files, ensure_ascii=False, sort_keys=True).encode('utf-8'), 9)
	return torch.frombuffer(bytearray(blob), dtype=torch.uint8)


def unpack_sources(tensor):
	return json.loads(zlib.decompress(tensor.numpy().tobytes()).decode('utf-8'))


def parse_dt(s):
	try:
		return datetime.fromisoformat(s)
	except (TypeError, ValueError):
		return None


# ----------------------------------------------------------------------------- run directory

def parse_settings(path):
	"""settings.txt = repr of the argparse.Namespace main.py wrote. Parsed with ast, keyword by keyword."""
	try:
		node = ast.parse(open(path, encoding='utf-8').read().strip(), mode='eval').body
		if not (isinstance(node, ast.Call) and getattr(node.func, 'attr', getattr(node.func, 'id', '')) == 'Namespace'):
			return None
	except Exception:
		return None
	out = {}
	for kw in node.keywords:
		try:
			out[kw.arg] = ast.literal_eval(kw.value)
		except Exception:
			pass
	return out or None


def nn_args_from_args(a):
	"""The dict main.py's run() gives to the net constructor, rebuilt from the Coach's args
	(--nn-opt KEY=VALUE included). None if an argument is missing."""
	if any(k not in a for k in NN_ARG_KEYS):
		return None
	nn = dict(lr=a['learn_rate'], dropout=a['dropout'], epochs=a['epochs'], batch_size=a['batch_size'],
	          nn_version=a['nn_version'], learn_rate=a['learn_rate'], no_compression=a['no_compression'],
	          q_weight=a['q_weight'])
	for kv in a.get('nn_opt') or []:
		key, sep, raw = str(kv).partition('=')
		if not sep:
			return None
		key, raw = key.strip(), raw.strip()
		if raw.lower() in ('true', 'false'):
			value = (raw.lower() == 'true')
		else:
			try:
				value = int(raw)
			except ValueError:
				try:
					value = float(raw)
				except ValueError:
					value = raw
		nn[key] = value
	return nn


def flat_sources(d, game):
	"""The flat code copy of a run directory, put back at its place in the tree. Returns (files, n_skipped)."""
	out, skipped = {}, 0
	for name in sorted(os.listdir(d)):
		p = os.path.join(d, name)
		if not (name.endswith('.py') and os.path.isfile(p) and os.path.getsize(p) > 0):
			continue
		if os.path.isfile(os.path.join(ROOT, name)):
			rel = name
		elif game and os.path.isfile(os.path.join(ROOT, game, name)):
			rel = f'{game}/{name}'
		else:
			skipped += 1
			continue
		try:
			with open(p, encoding='utf-8') as f:
				out[rel] = f.read()
		except (OSError, UnicodeDecodeError):
			skipped += 1
	return out, skipped


def run_directory(src, saved_at, notes):
	"""(settings dict, settings dict even if restarted, run dir) -- the first one is None when the run
	directory cannot be trusted for this file."""
	d = os.path.dirname(os.path.abspath(src))
	path = os.path.join(d, 'settings.txt')
	if not os.path.isfile(path):
		return None, None, None
	settings = parse_settings(path)
	if settings is None:
		notes.append('settings.txt not understood: ignored')
		return None, None, None
	if glob.glob(os.path.join(d, 'settings.[0-9]*')):             # the run was restarted in this directory
		if saved_at is None or saved_at.timestamp() < os.path.getmtime(path):
			notes.append('run restarted here after this file was saved: settings.txt and code copy ignored')
			return None, settings, None
	return settings, settings, d


# ----------------------------------------------------------------------------- conversion

def classify(old):
	prov = old.get('provenance')
	if 'full_model' in old:
		return 'old'
	if isinstance(prov, dict) and 'nn_args' in prov and 'nn_args' not in old:
		return 'previous'
	if 'nn_args' in old and isinstance(prov, dict):
		return 'new'
	return None


def convert(src, dst, force, game_cli, env):
	"""Returns a one-line report; raises ValueError with the reason when the file is refused."""
	if os.path.exists(dst) and not force:
		raise ValueError(f'{dst} already exists (use --force)')
	try:
		old = read_old(src)
	except (EOFError, zipfile.BadZipFile) as e:
		raise ValueError(f'unreadable or truncated file ({type(e).__name__})')
	except RuntimeError as e:
		if re.search(r'PytorchStreamReader|central directory|unexpected EOF|invalid magic', str(e), re.I):
			raise ValueError('unreadable or truncated file')
		raise
	if not isinstance(old, dict) or 'state_dict' not in old:
		raise ValueError('not a framework checkpoint (no state_dict)')
	kind = classify(old)
	if kind == 'new':
		raise ValueError('already in the new format')
	if kind is None:
		raise ValueError('neither old format (full_model) nor previous layout (provenance.nn_args)')
	sd = old['state_dict']
	if not (isinstance(sd, dict) and sd and all(torch.is_tensor(v) for v in sd.values())):
		raise ValueError('state_dict is empty or not a dict of tensors')
	notes = []
	mtime = datetime.fromtimestamp(os.path.getmtime(src)).astimezone().isoformat(timespec='seconds')
	old_prov = old.get('provenance') if isinstance(old.get('provenance'), dict) else {}

	# --- interface and version, from the file
	if kind == 'old':
		md = dict(vars(old['full_model']))

		def need(name, lo):
			if not is_int(md.get(name), lo):
				raise ValueError(f'model.{name} is {md.get(name)!r}')
			return int(md[name])

		version, actions, players = need('version', 1), need('action_size', 1), need('num_players', 2)
		if is_int(md.get('nb_vect'), 1) and is_int(md.get('vect_dim'), 1):
			board = [int(md['nb_vect']), int(md['vect_dim'])]
		elif md.get('board_size') is not None:
			board = shape_list(md['board_size'])
		else:
			raise ValueError('model has neither nb_vect+vect_dim nor board_size')
		own_nn_args = plain(dict(md['args']))[0] if isinstance(md.get('args'), dict) else None
		own_is_real = own_nn_args is not None
		extras, n_repr = plain({k: v for k, v in old.items() if k not in REMOVED_OLD})
		if 'nn_version' in old and old['nn_version'] != version:
			notes.append(f'root nn_version {old["nn_version"]!r} != model.version {version}; kept model.version')
		converted = f'{now_iso()} from old format; old file mtime {mtime}'
		saved_at_s = old_prov.get('saved_at') or mtime
	else:
		na = old_prov.get('nn_args')
		if not (isinstance(na, dict) and is_int(na.get('nn_version'), 1)):
			raise ValueError('provenance.nn_args has no int nn_version')
		version, own_nn_args = int(na['nn_version']), plain(dict(na))[0]
		actions, players = old_prov.get('action_size'), old_prov.get('num_players')
		board = old_prov.get('board_size')
		if not is_int(actions, 1):
			raise ValueError(f'provenance.action_size is {actions!r}')
		if not is_int(players, 2):
			raise ValueError(f'provenance.num_players is {players!r}')
		if not (isinstance(board, (list, tuple)) and board and all(is_int(x, 1) for x in board)):
			raise ValueError(f'provenance.board_size is {board!r}')
		actions, players, board = int(actions), int(players), [int(x) for x in board]
		own_is_real = 'converted' not in old_prov                # a converted file may carry default nn_args
		extras, n_repr = plain({k: v for k, v in old.items() if k not in STANDARD_PREVIOUS})
		converted = old_prov.get('converted')
		m = re.search(r'old file mtime (\S+)', converted) if isinstance(converted, str) else None
		saved_at_s = old_prov.get('saved_at') or (m.group(1) if m else None)
	if n_repr:
		notes.append(f'{n_repr} non-plain value(s) in the args stored as repr strings')
	extras.pop('nn_version', None)
	saved_at = parse_dt(saved_at_s)

	# --- run directory (settings.txt, flat code copy)
	settings, settings_any, run_dir = run_directory(src, saved_at, notes)
	known = {**(settings or {}), **extras}                       # the file's own args win
	known.setdefault('nn_version', version)                      # the Coach's args never carry it in the file
	args = dict(extras)
	for k, v in (settings or {}).items():
		if k != 'nn_version' and k not in args:
			args[k] = plain(v)[0]

	# --- game
	game = old_prov.get('game') or extras.get('game') or (settings_any or {}).get('game') or game_cli
	game = game if isinstance(game, str) and game else None
	if game is None:
		notes.append('no game name: pass --game')

	# --- nn_args
	rebuilt = nn_args_from_args(known)
	if rebuilt is not None and rebuilt['nn_version'] != version:
		notes.append(f'args say nn_version {rebuilt["nn_version"]} but the net is V{version}: args not used for nn_args')
		rebuilt = None
	if own_is_real and own_nn_args is not None:
		nn_args = own_nn_args
	elif rebuilt is not None:
		nn_args = rebuilt
		notes.append('nn_args rebuilt from the Coach args' + (' (settings.txt)' if settings else ''))
	elif own_nn_args is not None:
		nn_args = own_nn_args
	else:
		nn_args = dict(DEFAULT_NN_ARGS)
		notes.append('no source for nn_args: defaults (optional-branch flags unknown)')
	nn_args['nn_version'] = version

	# --- provenance
	prov = {'saved_at': saved_at_s, 'command_line': old_prov.get('command_line') if isinstance(old_prov.get('command_line'), str) else None}
	for k in ENV_VERSIONS:
		prov[k] = old_prov.get(k) or env.get(k)
	if converted:
		prov['converted'] = converted

	# --- sources
	sources = None
	if kind == 'previous' and isinstance(old.get('sources'), dict) and old['sources']:
		sources = {k: v for k, v in old['sources'].items() if isinstance(k, str) and isinstance(v, str)}
	elif run_dir and game:
		sources, skipped = flat_sources(run_dir, game)
		notes.append(f'sources from the run directory: {len(sources)} file(s)' + (f', {skipped} skipped' if skipped else ''))
	elif run_dir:
		notes.append('run directory found but no game name: sources not rebuilt')

	new = {'state_dict': sd, 'game': game, 'board_size': board, 'action_size': actions, 'num_players': players,
	       'nn_args': nn_args, 'args': args, 'provenance': prov}
	if sources:
		new['sources'] = pack_sources(sources)

	tmp = dst + '.tmp'
	os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
	try:
		torch.save(new, tmp)
		back = torch.load(tmp, map_location='cpu', weights_only=True)      # must be loadable without any code
		if set(back['state_dict']) != set(sd) or not all(same_tensor(back['state_dict'][k], sd[k]) for k in sd):
			raise ValueError('re-read state_dict differs from the original')
		if back['nn_args']['nn_version'] != version:
			raise ValueError('re-read nn_args differs')
		if sources and unpack_sources(back['sources']) != sources:
			raise ValueError('re-read sources differ')
		os.replace(tmp, dst)
	finally:
		if os.path.exists(tmp):
			os.remove(tmp)
	return f'V{version}, P={players}, in={board}, actions={actions}, {len(sd)} tensors, game {game or "-"}' + \
	       (f', {len(sources)} source file(s)' if sources else '') + (f'  [{"; ".join(notes)}]' if notes else '')


def main():
	ap = argparse.ArgumentParser(description='Convert old-format / previous-layout checkpoints to the current layout.')
	ap.add_argument('inputs', nargs='+', help='.pt files')
	ap.add_argument('-o', '--out', required=True, help='output file (one input) or output directory; the input itself is allowed with --force')
	ap.add_argument('--force', action='store_true', help='overwrite existing outputs')
	ap.add_argument('--game', default=None, help='game name, used only when the file and its settings.txt have none')
	for k in ENV_VERSIONS:
		ap.add_argument(f'--{k}', default=None, help=f'{k} version recorded when the file does not have one (default: ENV_VERSIONS in the script)')
	a = ap.parse_args()
	env = {k: getattr(a, k) or ENV_VERSIONS[k] for k in ENV_VERSIONS}

	as_dir = len(a.inputs) > 1 or os.path.isdir(a.out) or a.out.endswith(('/', os.sep))
	if len(a.inputs) > 1 and os.path.exists(a.out) and not os.path.isdir(a.out):
		ap.error('several inputs: --out must be a directory')
	names = [os.path.basename(p) for p in a.inputs]
	if as_dir and len(set(names)) != len(names):
		ap.error('two inputs have the same file name: convert them separately')

	failed = 0
	for src, name in zip(a.inputs, names):
		dst = os.path.join(a.out, name) if as_dir else a.out
		try:
			print(f'  OK    {src} -> {dst}: {convert(src, dst, a.force, a.game, env)}', flush=True)
		except Exception as e:
			failed += 1
			print(f'  FAIL  {src}: {e if isinstance(e, ValueError) else f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"}',
			      flush=True)
	print(f'\n{len(a.inputs) - failed}/{len(a.inputs)} converted')
	sys.exit(2 if failed else 0)


if __name__ == '__main__':
	main()
