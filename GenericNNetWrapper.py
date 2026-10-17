import os
import sys
import time
import pickle
import zlib
import importlib.metadata
import inspect
import json
import platform
import shlex
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "1" # PyTorch more efficient this way

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
import torch.onnx
import onnxruntime as ort
import onnx
torch.set_num_threads(1) # PyTorch more efficient this way


# Checkpoint layout: plain python types + tensors only, so that torch.load(weights_only=True)
# reads it without any game code.
#
#     state_dict
#     game          name accepted by GameSwitcher.import_game (folder of the game package)
#     board_size    list of int          (game.getBoardSize())
#     action_size   int
#     num_players   int
#     nn_args       dict given to the net constructor (nn_version is HERE only)
#     args          what the Coach passes as additional_keys (vars(args) of main.py), without nn_version
#     provenance    saved_at      ISO-8601 with UTC offset, refreshed at each save
#                   iteration     training iteration of the Coach loop (None if unknown; temp.pt = i-1, the
#                                 state before training iteration i; restarts at 1 on a relaunch)
#                   command_line  sys.argv of the process that trained
#                   python, torch, numpy, numba, onnx, onnxruntime   versions in use
#     sources       uint8 tensor = zlib(JSON {path relative to the framework root: text}) of the *.py
#                   files of the framework root and of the game package (not recursive, like the code
#                   backup of main.py); unpack_sources() gives the dict back
#
# Everything but state_dict, args, saved_at and iteration is captured ONCE, when a network wrapper is built
# (a long run outlives edits of the sources: record what the process actually imported).

ROOT = os.path.dirname(os.path.realpath(__file__))


def now_iso():
	return datetime.now().astimezone().isoformat(timespec='seconds')


def _plain(x):
	"""Recursively convert to builtin types, so that weights_only=True can load the result."""
	if isinstance(x, dict):
		return {str(k): _plain(v) for k, v in x.items()}
	if isinstance(x, (list, tuple)):
		return [_plain(v) for v in x]
	if hasattr(x, 'item') and getattr(x, 'ndim', None) == 0:   # numpy / torch scalar
		return x.item()
	if x is None or isinstance(x, bool):
		return x
	for t in (int, float, str):   # exact builtin type: subclasses (TorchVersion, enums...) do not load with weights_only
		if isinstance(x, t):
			return t(x)
	return repr(x)


def shape_list(x):
	"""getBoardSize() may return an int or a tuple: always a list of int."""
	return [int(d) for d in (x if isinstance(x, (list, tuple)) else [x])]


def game_name(wrapper):
	"""The game package folder (direct subdirectory of the framework root) holding the wrapper's
	file, i.e. the name GameSwitcher.import_game() takes. None if the wrapper lives elsewhere."""
	try:
		d = os.path.dirname(os.path.realpath(inspect.getsourcefile(type(wrapper))))
		return os.path.basename(d) if os.path.dirname(d) == ROOT else None
	except Exception:
		return None


def lib_version(name):
	try:
		return importlib.metadata.version(name)
	except Exception:
		return None


def collect_sources(game):
	"""{relative path: text} of the *.py files of the framework root and of the game package
	(not recursive, empty files skipped)."""
	out = {}
	for d in [ROOT] + ([os.path.join(ROOT, game)] if game else []):
		for name in sorted(os.listdir(d)):
			p = os.path.join(d, name)
			if not (name.endswith('.py') and os.path.isfile(p) and os.path.getsize(p) > 0):
				continue
			try:
				with open(p, encoding='utf-8') as fh:
					out[os.path.relpath(p, ROOT).replace(os.sep, '/')] = fh.read()
			except (OSError, UnicodeDecodeError):
				pass
	return out


def pack_sources(files):
	"""{path: text} -> uint8 tensor holding zlib(JSON). A tensor weighs exactly its size in the .pt
	(bytes weigh 1.5x, base64 1.33x)."""
	blob = zlib.compress(json.dumps(files, ensure_ascii=False, sort_keys=True).encode('utf-8'), 9)
	return torch.frombuffer(bytearray(blob), dtype=torch.uint8)


def unpack_sources(tensor):
	return json.loads(zlib.decompress(tensor.numpy().tobytes()).decode('utf-8'))


def capture_header(wrapper):
	"""Everything a checkpoint records besides the weights and the Coach's args.
	Never raises: the optional parts degrade to None / absent."""
	game = game_name(wrapper)
	header = {
		'game': game,
		'board_size': shape_list(wrapper.board_size),
		'action_size': int(wrapper.action_size),
		'num_players': int(wrapper.num_players),
		'nn_args': _plain(dict(wrapper.args)),
		'provenance': {
			'saved_at': None,
			'command_line': None,
			'python': platform.python_version(),
			'torch': str(torch.__version__),
			'numpy': lib_version('numpy'),
			'numba': lib_version('numba'),
			'onnx': lib_version('onnx'),
			'onnxruntime': lib_version('onnxruntime'),
		},
	}
	try:
		header['provenance']['command_line'] = shlex.join(sys.argv)
	except Exception:
		pass
	try:
		header['sources'] = pack_sources(collect_sources(game))
	except Exception as e:
		print(f'[provenance] sources not recorded: {e!r}')
	return header


def checkpoint_version(ckpt):
	"""nn_version of a loaded checkpoint (single source of truth: nn_args)."""
	try:
		return ckpt['nn_args']['nn_version']
	except (KeyError, TypeError):
		raise ValueError('old-format checkpoint (no nn_args.nn_version): '
		                 'convert it first with convert_checkpoint.py') from None



class GenericNNetWrapper(NeuralNet):
	def __init__(self, game, nn_args):
		self.args = nn_args
		self.game = game   # to rebuild the net when an additive-compatible load is refused
		self.device = {
			'training' : 'cpu', #'cuda' if torch.cuda.is_available() else 'cpu',
			'inference': 'onnx',
			'just_loaded': 'cpu',
		}
		self.current_mode = 'cpu'
		self.init_nnet(game, nn_args)
		self.ort_session = None

		self.board_size = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.requestKnowledgeTransfer = False
		# frozen NOW: a long run outlives edits of the sources, record what was actually imported
		self.header = capture_header(self)

	def init_nnet(self, game, nn_args):
		pass

	def train(self, examples):
		"""
		examples: list of examples, each example is of form (board, pi, v)
		"""
		self.switch_target('training')
		optimizer = optim.AdamW(self.nnet.parameters(), lr=self.args['learn_rate'])
		batch_count = int(len(examples) / self.args['batch_size'])
		scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args['learn_rate'], steps_per_epoch=batch_count, epochs=self.args['epochs'])

		t = tqdm(total=self.args['epochs'] * batch_count, desc='Train ep0', colour='blue', ncols=120, mininterval=0.5, disable=None)
		for epoch in range(self.args['epochs']):
			t.set_description(f'Train ep{epoch + 1}')
			self.nnet.train()
			pi_losses, v_losses = AverageMeter(), AverageMeter()
	
			for i_batch in range(batch_count):
				sample_ids = np.random.choice(len(examples), size=self.args['batch_size'], replace=False)
				boards, pis, vs, valid_actions, qs = self.pick_examples(examples, sample_ids)
				boards = torch.FloatTensor(np.array(boards).astype(np.float32))
				valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_))
				target_pis = torch.FloatTensor(np.array(pis).astype(np.float32))
				target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))
				target_qs = torch.FloatTensor(np.array(qs).astype(np.float32))

				# predict
				optimizer.zero_grad(set_to_none=True)
				out_pi, out_v = self.nnet(boards, valid_actions)
				l_pi, l_v = self.loss_pi(target_pis, out_pi), self.loss_v(target_vs, target_qs, out_v)
				total_loss = l_pi + 0.25*l_v # Weight 0.25 * value

				# record loss
				pi_losses.update(l_pi.item(), boards.size(0))
				v_losses.update(l_v.item(), boards.size(0))
				t.set_postfix(PI=pi_losses, V=v_losses, refresh=False)

				# compute gradient and do SGD step
				total_loss.backward()
				optimizer.step()
				scheduler.step()

				t.update()

		t.close()
		
	def predict(self, board, valid_actions):
		"""
		board: np array with board
		"""
		self.switch_target('inference')

		if self.current_mode == 'onnx':
			ort_outs = self.ort_session.run(None, {
				'board': np.expand_dims(board.astype(np.float32), 0),
				'valid_actions': np.expand_dims(np.array(valid_actions).astype(np.bool_), 0),
			})
			pi, v = np.exp(ort_outs[0][0]), ort_outs[1][0]
			return pi, v

		else:
			board = torch.FloatTensor(board.astype(np.float32)).unsqueeze(0)
			valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_)).unsqueeze(0)
			if self.current_mode == 'cuda':
				board, valid_actions = board.contiguous().cuda(), valid_actions.contiguous().cuda()
			self.nnet.eval()
			with torch.no_grad():
				pi, v = self.nnet(board, valid_actions)
			pi, v = torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
			return pi, v

	def predict_client(self, board, valid_actions, batch_info):
		if self.current_mode != 'onnx':
			raise Exception('Batch prediction only in ONNX mode')

		i_thread, i_result, shared_memory, locks = batch_info

		# Store inputs in shared memory
		shared_memory[i_thread] = (
			np.expand_dims(board.astype(np.float32), 0),
			np.expand_dims(np.array(valid_actions).astype(np.bool_), 0),
		)
		# Unblock next thread (= next MCTS or server), and wait for our turn
		locks[i_thread+1].release()
		locks[i_thread].acquire()

		# Retrieve results in shared memory
		ort_outs = shared_memory[i_result]
		pi, v = np.exp(ort_outs[0]), ort_outs[1]

		return pi, v

	def predict_server(self, nb_threads, shared_memory, locks):
		self.switch_target('inference')
		locks[0].release()

		while shared_memory[-1] <= 1:
			locks[-1].acquire() # Wait for all inputs

			ort_outs = self.ort_session.run(None, {
				'board'        : np.concatenate([shared_memory[i][0] for i in range(nb_threads)]),
				'valid_actions': np.concatenate([shared_memory[i][1] for i in range(nb_threads)]),
			})
			for i in range(nb_threads):
				shared_memory[i+nb_threads] = (ort_outs[0][i], ort_outs[1][i])

			locks[0].release() # Unblock 1st thread

	def loss_pi(self, targets, outputs):
		loss_ = torch.nn.KLDivLoss(reduction="batchmean")
		return loss_(outputs, targets)

	def loss_v(self, targets_V, targets_Q, outputs):
		targets = (targets_V + self.args['q_weight'] * targets_Q) / (1+self.args['q_weight'])
		return torch.sum((targets - outputs) ** 2) / (targets_V.size()[0] * targets_V.size()[-1]) # Normalize by batch size * nb of players

	def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', additional_keys={}, iteration=None):
		filepath = os.path.join(folder, filename)
		if not os.path.exists(folder):
			os.mkdir(folder)

		h = self.header
		data = {
			'state_dict': self.nnet.state_dict(),
			'game': h['game'],
			'board_size': h['board_size'],
			'action_size': h['action_size'],
			'num_players': h['num_players'],
			'nn_args': h['nn_args'],
			# nn_version lives in nn_args only (vars(args) would add a duplicate)
			'args': _plain({k: v for k, v in additional_keys.items() if k != 'nn_version'}),
			'provenance': dict(h['provenance'], saved_at=now_iso(), iteration=None if iteration is None else int(iteration)),
		}
		if 'sources' in h:
			data['sources'] = h['sources']
		torch.save(data, filepath)

	def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
		# Fail loudly: a silent fallback would train or play a random network
		filepath = os.path.join(folder, filename)
		if not os.path.exists(filepath):
			raise FileNotFoundError(f'No model in path {filepath}')
		checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
		self.load_network(checkpoint)
		self.switch_target('just_loaded')
		return checkpoint

	def load_network(self, checkpoint):
		ckpt_version = checkpoint_version(checkpoint)   # raises on an old-format file: convert_checkpoint.py

		if ckpt_version != self.args['nn_version']:
			print('Checkpoint includes NN version', ckpt_version, ', but you ask version', self.args['nn_version'], ' so not loading it and initiate knowledge transfer')
			self.requestKnowledgeTransfer = True
			return

		try:
			self.nnet.load_state_dict(checkpoint['state_dict'])
		except RuntimeError as e:
			# Same version, but checkpoint written before some purely ADDITIVE,
			# zero-initialised tensors existed: accept it iff nothing else differs,
			# so the loaded net computes exactly the checkpoint's function.
			if self._load_additive_compatible(checkpoint['state_dict'], ckpt_version):
				return
			print(f'Cant load NN {ckpt_version} in checkpoint ({e}), so initiate knowledge transfer')
			self.requestKnowledgeTransfer = True

	def _load_additive_compatible(self, state_dict, ckpt_version):
		prefixes = getattr(self.nnet, 'additive_param_prefixes', ())
		if not prefixes or ckpt_version != self.nnet.version:
			return False
		try:
			result = self.nnet.load_state_dict(state_dict, strict=False)  # raises on shape mismatch
		except Exception as e:
			print(f'additive-compatible load refused (shape mismatch): {e}')
			self.init_nnet(self.game, self.args)   # no half-loaded net
			return False
		if result.unexpected_keys or not all(k.startswith(prefixes) for k in result.missing_keys):
			print(f'additive-compatible load refused: unexpected={result.unexpected_keys} missing={result.missing_keys}')
			self.init_nnet(self.game, self.args)
			return False
		self.nnet.version = ckpt_version
		return True

	def switch_target(self, mode):
		target_device = self.device[mode]
		if target_device == self.current_mode:
			return

		if target_device == 'cpu':
			self.nnet.cpu()
			torch.cuda.empty_cache()
			self.ort_session = None # Make ONNX export invalid
		elif target_device == 'onnx':
			self.nnet.cpu()
			self.export_and_load_onnx()
		elif target_device == 'cuda':
			self.nnet.cuda()
			self.ort_session = None # Make ONNX export invalid
		elif target_device == 'just_loaded':
			self.ort_session = None # Make ONNX export invalid
		
		self.current_mode = target_device

	def export_and_load_onnx(self):
		dummy_board         = torch.randn(self.board_size, dtype=torch.float32).unsqueeze(0)
		dummy_valid_actions = torch.BoolTensor(torch.randn(self.action_size)>0.5).unsqueeze(0)
		self.nnet.to('cpu')
		self.nnet.eval()

		temporary_file = 'nn_export_' + str( int(time.time()*1000)%1000000 ) + '.onnx'
		torch.onnx.export(
			self.nnet,
			(dummy_board, dummy_valid_actions),
			temporary_file,
			input_names = ['board', 'valid_actions'],
			output_names = ['pi', 'v'],
			dynamic_axes={
				'board'        : {0: 'batch_size'},
				'valid_actions': {0: 'batch_size'},
				'pi'           : {0: 'batch_size'},
				'v'            : {0: 'batch_size'},
			}
		)
		if ort.__version__ >= '1.17.0':
			# Convert ONNX file to most recent opset version
			model_with_old_opset = onnx.load(temporary_file)
			model_with_new_opset = onnx.version_converter.convert_version(model_with_old_opset, 21)
			onnx.save(model_with_new_opset, temporary_file)

		opts = ort.SessionOptions()
		opts.intra_op_num_threads = 1   # inter-op threads and sequential mode are onnxruntime defaults
		self.ort_session = ort.InferenceSession(temporary_file, sess_options=opts, providers=['CPUExecutionProvider'])
		os.remove(temporary_file)

	def pick_examples(self, examples, sample_ids):
		if self.args['no_compression']:
			picked_examples = [examples[i] for i in sample_ids]
		else: 
			picked_examples = [pickle.loads(zlib.decompress(examples[i])) for i in sample_ids]
		return list(zip(*picked_examples))
	
	def reshape_boards(self, numpy_boards):
		# Some game needs to reshape boards before being an input of NNet
		return numpy_boards

	def number_params(self):
		total_params = sum(p.numel() for p in self.nnet.parameters())
		trainable_params = sum(p.numel() for p in self.nnet.parameters() if p.requires_grad)
		return total_params, trainable_params

if __name__ == "__main__":
	# Inspection tool: architecture cost (MFlops, parameters) and checkpoint metadata
	import argparse
	from GameSwitcher import import_game

	parser = argparse.ArgumentParser(description='NNet inspector')
	parser.add_argument('game'               , action='store', help='The name of the game')
	parser.add_argument('--input'      , '-i', action='store', default=None , help='Checkpoint to inspect')
	parser.add_argument('--nn-version' , '-V', action='store', default=None , type=int, help='Architecture to build (default: the one of --input)')
	args = parser.parse_args()
	if args.input is None and args.nn_version is None:
		raise SystemExit('Specify a checkpoint (--input) and/or an architecture (-V)')
	Game, NNet, players, NUMBER_PLAYERS = import_game(args.game)

	checkpoint = torch.load(args.input, map_location='cpu', weights_only=False) if args.input else {}
	if args.nn_version is None:
		args.nn_version = checkpoint_version(checkpoint)

	g = Game()
	nn_args = dict(lr=None, dropout=0., epochs=None, batch_size=None, nn_version=args.nn_version,
				   learn_rate=None, no_compression=False, q_weight=0.)
	nnet = NNet(g, nn_args)
	if args.input:
		nnet.load_checkpoint(os.path.dirname(args.input), os.path.basename(args.input))

	from fvcore.nn import FlopCountAnalysis
	dummy_board         = torch.randn(g.getBoardSize(), dtype=torch.float32).unsqueeze(0)
	dummy_valid_actions = torch.BoolTensor(torch.randn(1, g.getActionSize())>0.5)
	nnet.nnet.eval()
	flops = FlopCountAnalysis(nnet.nnet, (dummy_board, dummy_valid_actions))
	flops.unsupported_ops_warnings(False)
	print(f'V{nnet.nnet.version} -> {flops.total()/1000000:.1f} MFlops, nb params {nnet.number_params()[0]:.2e}')

	for k in sorted(checkpoint.keys()):
		if k not in ['state_dict', 'sources']:
			print(f'  {k}: {checkpoint[k]}')
	print(f'Board shape: {list(dummy_board.shape)}, valids shape: {list(dummy_valid_actions.shape)}')
