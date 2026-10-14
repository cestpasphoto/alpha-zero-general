#!/usr/bin/env python3
"""
convert_checkpoint.py -- OLD-format checkpoint (pickled full_model) -> NEW format (provenance).

  python convert_checkpoint.py OLD.pt -o NEW.pt
  python convert_checkpoint.py a.pt b.pt c.pt -o OUT_DIR/ [--force]

Run from the framework root. An existing output (the input file itself included) is never
overwritten without --force; the replacement is atomic and checked first, rely on git to go back. No game rule is executed: classes that cannot be imported are read
through stubs, only attributes and tensors are used.

What is built (layout: see GenericNNetWrapper.py)
  state_dict      copied unchanged
  provenance      saved_at / command_line / git / torch: carried over from an old 'provenance' dict
                  if it had them, otherwise None (unknown, never invented)
                  board_size   [nb_vect, vect_dim] or the model's board_size
                  action_size, num_players   from the pickled model
                  nn_args      model.args if the model kept it (Catan does not: defaults + nn_version)
                  nn_args['nn_version'] = model.version (the net itself is the authority; a different
                  root 'nn_version', overwritten by the Coach's vars(args), is reported)
                  converted    when this file was converted from the old format, and the old file's mtime
  sources         ABSENT: the code of that time is unknown. verify_checkpoints.py does not warn about
                  it for converted files. Never fill it with today's files: it would be a false record.
  game            old provenance.game, else the root 'game' key (vars(args) of the Coach), else None
  other root keys kept as they are (vars(args) of the Coach, optim_state), except the removed
                  full_model / nn_version / onnx_parity_skipped.

Each output is re-read with torch.load(weights_only=True) and compared tensor by tensor before it
replaces the temporary file. Exit code 0 if every file was converted, 2 otherwise.
"""

import argparse
import os
import pickle
import re
import sys
import types
import zipfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch



def now_iso():
	return datetime.now().astimezone().isoformat(timespec='seconds')


def shape_list(x):
	return [int(d) for d in (x if isinstance(x, (list, tuple)) else [x])]

REMOVED = {'full_model', 'nn_version', 'onnx_parity_skipped', 'state_dict', 'provenance'}
DEFAULT_NN_ARGS = dict(lr=None, dropout=0., epochs=None, batch_size=None, learn_rate=None,
                       no_compression=False, q_weight=0.)


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


def convert(src, dst, force):
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
	if 'full_model' not in old:
		if isinstance(old.get('provenance'), dict) and 'nn_args' in old['provenance']:
			raise ValueError('already in the new format')
		raise ValueError('no full_model: nothing to read the architecture from')
	sd = old['state_dict']
	if not (isinstance(sd, dict) and sd and all(torch.is_tensor(v) for v in sd.values())):
		raise ValueError('state_dict is empty or not a dict of tensors')

	md = dict(vars(old['full_model']))
	notes = []

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

	args = md.get('args')
	if isinstance(args, dict):
		nn_args, _ = plain(dict(args))
	else:
		nn_args = dict(DEFAULT_NN_ARGS)
		notes.append('no model.args: default nn_args (optional-branch flags unknown)')
	if nn_args.get('nn_version') not in (None, version):
		notes.append(f'model.args.nn_version {nn_args["nn_version"]} != model.version {version}')
	nn_args['nn_version'] = version
	if 'nn_version' in old and old['nn_version'] != version:
		notes.append(f'root nn_version {old["nn_version"]!r} != model.version {version} (overwritten by vars(args)); kept model.version')

	old_prov = old.get('provenance') if isinstance(old.get('provenance'), dict) else {}
	prov = {'saved_at': None, 'command_line': None, 'git': {'commit': None, 'dirty': None}, 'torch': None}
	prov.update({k: plain(old_prov[k])[0] for k in ('saved_at', 'command_line', 'git', 'torch') if k in old_prov})
	game = old_prov.get('game', old.get('game'))                 # root 'game' = the Coach's vars(args), when it was saved
	prov['game'] = game if isinstance(game, str) and game else None
	if prov['game'] is None:
		notes.append('no game name found: pass it to verify_checkpoints.py / pit.py')
	mtime = datetime.fromtimestamp(os.path.getmtime(src)).astimezone().isoformat(timespec='seconds')
	prov.update(board_size=board, action_size=actions, num_players=players, nn_args=nn_args,
	            converted=f'{now_iso()} from old format; old file mtime {mtime}')

	extras, n_repr = plain({k: v for k, v in old.items() if k not in REMOVED})
	if n_repr:
		notes.append(f'{n_repr} non-plain value(s) in other keys stored as repr strings')
	new = {'state_dict': sd, 'provenance': prov, **extras}

	tmp = dst + '.tmp'
	os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
	try:
		torch.save(new, tmp)
		back = torch.load(tmp, map_location='cpu', weights_only=True)      # must be loadable without any code
		if set(back['state_dict']) != set(sd) or not all(same_tensor(back['state_dict'][k], sd[k]) for k in sd):
			raise ValueError('re-read state_dict differs from the original')
		if back['provenance']['nn_args']['nn_version'] != version:
			raise ValueError('re-read provenance differs')
		os.replace(tmp, dst)
	finally:
		if os.path.exists(tmp):
			os.remove(tmp)
	return f'V{version}, P={players}, in={board}, actions={actions}, {len(sd)} tensors' + \
	       (f'  [{"; ".join(notes)}]' if notes else '')


def main():
	ap = argparse.ArgumentParser(description='Convert old-format checkpoints (full_model) to the provenance format.')
	ap.add_argument('inputs', nargs='+', help='old-format .pt files')
	ap.add_argument('-o', '--out', required=True, help='output file (one input) or output directory')
	ap.add_argument('--force', action='store_true', help='overwrite existing outputs')
	a = ap.parse_args()

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
			print(f'  OK    {src} -> {dst}: {convert(src, dst, a.force)}', flush=True)
		except Exception as e:
			failed += 1
			print(f'  FAIL  {src}: {e if isinstance(e, ValueError) else f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"}',
			      flush=True)
	print(f'\n{len(a.inputs) - failed}/{len(a.inputs)} converted')
	sys.exit(2 if failed else 0)


if __name__ == '__main__':
	main()
#!/usr/bin/env python3
"""
convert_checkpoint.py -- OLD-format checkpoint (pickled full_model) -> NEW format (provenance).

  python convert_checkpoint.py OLD.pt -o NEW.pt
  python convert_checkpoint.py a.pt b.pt c.pt -o OUT_DIR/ [--force]

Run from the framework root. Originals are never modified; an existing output is never
overwritten without --force. No game rule is executed: classes that cannot be imported are read
through stubs, only attributes and tensors are used.

What is built (layout: see GenericNNetWrapper.py)
  state_dict      copied unchanged
  provenance      saved_at / command_line / git / torch: carried over from an old 'provenance' dict
                  if it had them, otherwise None (unknown, never invented)
                  board_size   [nb_vect, vect_dim] or the model's board_size
                  action_size, num_players   from the pickled model
                  nn_args      model.args if the model kept it (Catan does not: defaults + nn_version)
                  nn_args['nn_version'] = model.version (the net itself is the authority; a different
                  root 'nn_version', overwritten by the Coach's vars(args), is reported)
                  converted    when this file was converted from the old format, and the old file's mtime
  sources         ABSENT: the code of that time is unknown. verify_checkpoints.py does not warn about
                  it for converted files. Never fill it with today's files: it would be a false record.
  other root keys kept as they are (vars(args) of the Coach, optim_state), except the removed
                  full_model / nn_version / onnx_parity_skipped.

Each output is re-read with torch.load(weights_only=True) and compared tensor by tensor before it
replaces the temporary file. Exit code 0 if every file was converted, 2 otherwise.
"""

import argparse
import os
import pickle
import re
import sys
import types
import zipfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch



def now_iso():
	return datetime.now().astimezone().isoformat(timespec='seconds')


def shape_list(x):
	return [int(d) for d in (x if isinstance(x, (list, tuple)) else [x])]

REMOVED = {'full_model', 'nn_version', 'onnx_parity_skipped', 'state_dict', 'provenance'}
DEFAULT_NN_ARGS = dict(lr=None, dropout=0., epochs=None, batch_size=None, learn_rate=None,
                       no_compression=False, q_weight=0.)


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


def convert(src, dst, force):
	"""Returns a one-line report; raises ValueError with the reason when the file is refused."""
	if os.path.exists(dst) and not force:
		raise ValueError(f'{dst} already exists (use --force)')
	if os.path.realpath(src) == os.path.realpath(dst):
		raise ValueError('output is the input file')
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
	if 'full_model' not in old:
		if isinstance(old.get('provenance'), dict) and 'nn_args' in old['provenance']:
			raise ValueError('already in the new format')
		raise ValueError('no full_model: nothing to read the architecture from')
	sd = old['state_dict']
	if not (isinstance(sd, dict) and sd and all(torch.is_tensor(v) for v in sd.values())):
		raise ValueError('state_dict is empty or not a dict of tensors')

	md = dict(vars(old['full_model']))
	notes = []

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

	args = md.get('args')
	if isinstance(args, dict):
		nn_args, _ = plain(dict(args))
	else:
		nn_args = dict(DEFAULT_NN_ARGS)
		notes.append('no model.args: default nn_args (optional-branch flags unknown)')
	if nn_args.get('nn_version') not in (None, version):
		notes.append(f'model.args.nn_version {nn_args["nn_version"]} != model.version {version}')
	nn_args['nn_version'] = version
	if 'nn_version' in old and old['nn_version'] != version:
		notes.append(f'root nn_version {old["nn_version"]!r} != model.version {version} (overwritten by vars(args)); kept model.version')

	old_prov = old.get('provenance') if isinstance(old.get('provenance'), dict) else {}
	prov = {'saved_at': None, 'command_line': None, 'git': {'commit': None, 'dirty': None}, 'torch': None}
	prov.update({k: plain(old_prov[k])[0] for k in ('saved_at', 'command_line', 'git', 'torch') if k in old_prov})
	mtime = datetime.fromtimestamp(os.path.getmtime(src)).astimezone().isoformat(timespec='seconds')
	prov.update(board_size=board, action_size=actions, num_players=players, nn_args=nn_args,
	            converted=f'{now_iso()} from old format; old file mtime {mtime}')

	extras, n_repr = plain({k: v for k, v in old.items() if k not in REMOVED})
	if n_repr:
		notes.append(f'{n_repr} non-plain value(s) in other keys stored as repr strings')
	new = {'state_dict': sd, 'provenance': prov, **extras}

	tmp = dst + '.tmp'
	os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
	try:
		torch.save(new, tmp)
		back = torch.load(tmp, map_location='cpu', weights_only=True)      # must be loadable without any code
		if set(back['state_dict']) != set(sd) or not all(same_tensor(back['state_dict'][k], sd[k]) for k in sd):
			raise ValueError('re-read state_dict differs from the original')
		if back['provenance']['nn_args']['nn_version'] != version:
			raise ValueError('re-read provenance differs')
		os.replace(tmp, dst)
	finally:
		if os.path.exists(tmp):
			os.remove(tmp)
	return f'V{version}, P={players}, in={board}, actions={actions}, {len(sd)} tensors' + \
	       (f'  [{"; ".join(notes)}]' if notes else '')


def main():
	ap = argparse.ArgumentParser(description='Convert old-format checkpoints (full_model) to the provenance format.')
	ap.add_argument('inputs', nargs='+', help='old-format .pt files')
	ap.add_argument('-o', '--out', required=True, help='output file (one input) or output directory')
	ap.add_argument('--force', action='store_true', help='overwrite existing outputs')
	a = ap.parse_args()

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
			print(f'  OK    {src} -> {dst}: {convert(src, dst, a.force)}', flush=True)
		except Exception as e:
			failed += 1
			print(f'  FAIL  {src}: {e if isinstance(e, ValueError) else f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"}',
			      flush=True)
	print(f'\n{len(a.inputs) - failed}/{len(a.inputs)} converted')
	sys.exit(2 if failed else 0)


if __name__ == '__main__':
	main()
