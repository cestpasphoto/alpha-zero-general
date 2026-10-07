import os
import sys
import time
import pickle
import zlib

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

	def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', additional_keys={}):
		filepath = os.path.join(folder, filename)
		if not os.path.exists(folder):
			os.mkdir(folder)

		data = {
			'state_dict': self.nnet.state_dict(),
			'full_model': self.nnet,
			'nn_version': self.nnet.version,   # explicit, do not rely on the pickled model
			'onnx_parity_skipped': os.environ.get('SKIP_ONNX_PARITY') == '1',
		}
		data.update(additional_keys)
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
		# explicit 'nn_version' key, or full_model.version for older checkpoints
		ckpt_version = checkpoint.get('nn_version', checkpoint['full_model'].version)

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
		# the function that PLAYS must be the function that was TRAINED; every
		# ONNX session (self-play, arena, pit) is born here
		if os.environ.get('SKIP_ONNX_PARITY') != '1':
			self._assert_onnx_parity(verbose=(os.environ.get('ONNX_PARITY_VERBOSE') == '1'))


	def _assert_onnx_parity(self, n_synth=256, batch_size=8, tol_pi=1e-4, tol_v=1e-4,
	                        seed=0, verbose=False, raise_on_fail=True):
		"""
		Compare the torch module and the freshly exported ONNX session. Raises
		RuntimeError on an INTEGER-OP divergence. Cost ~0.3 s per export.

		Synthetic boards come in PAIRS: a signed board and its abs() twin. Integer
		ops that diverge between torch and ONNX (floor vs truncating division)
		only touch negative values, so they show as signed >> abs, already at
		small amplitude (-1 // 2). Uniform int8 noise is out of distribution for
		a trained net: absolute errors there can exceed 1e-4 through plain float32
		conditioning (e.g. BatchNorm with tiny running_var), equally on both twins;
		that case is reported, not raised. Real-position parity is checked by
		check_onnx_parity.py. Batch 8 and batch 1 are both tested, since
		predict_server() runs the graph batched.
		"""
		import numpy as _np
		rng = _np.random.default_rng(seed)
		base = rng.integers(-128, 128, size=(n_synth,) + tuple(self.board_size)).astype(_np.float32)
		valids = rng.random((n_synth, self.action_size)) > 0.5
		valids[:, 0] = True                                       # no all-illegal row

		was_training = self.nnet.training
		self.nnet.eval()

		def errors(boards, bs):
			with torch.no_grad():
				pi_t, v_t = self.nnet(torch.from_numpy(boards), torch.from_numpy(valids))
			pi_t, v_t = pi_t.numpy(), v_t.numpy()
			pi_o, v_o = _np.empty_like(pi_t), _np.empty_like(v_t)
			for s in range(0, n_synth, bs):
				e = min(s + bs, n_synth)
				out = self.ort_session.run(None, {'board': boards[s:e], 'valid_actions': valids[s:e]})
				pi_o[s:e], v_o[s:e] = out[0], out[1]
			dpi = _np.where(valids, _np.abs(_np.exp(pi_t) - _np.exp(pi_o)), 0.).max(axis=1)
			dv = _np.abs(v_t - v_o).reshape(n_synth, -1).max(axis=1)
			return dpi, dv

		failures, warnings_ = [], []
		for amp in (8, 128):
			signed = _np.clip(_np.round(base * amp / 128.), -amp, amp - 1).astype(_np.float32)
			for bs in (batch_size, 1):
				dpi_s, dv_s = errors(signed, bs)
				dpi_a, dv_a = errors(_np.abs(signed), bs)
				err_s, err_a = _np.maximum(dpi_s / tol_pi, dv_s / tol_v), _np.maximum(dpi_a / tol_pi, dv_a / tol_v)
				sign_bias = err_s.max() > 1 and _np.median(err_s) > 10 * _np.median(err_a) + 1e-3
				line = (f'[onnx-parity] amp={amp:3d} bs={bs}  signed: max|dpi|={dpi_s.max():.2e} max|dv|={dv_s.max():.2e}'
				        f'  abs twin: max|dpi|={dpi_a.max():.2e} max|dv|={dv_a.max():.2e}')
				if sign_bias:
					print(line + '  *** SIGN-BIASED MISMATCH ***')
					failures.append((amp, bs))
				elif max(err_s.max(), err_a.max()) > 1:
					warnings_.append(line + '  (over tol on BOTH twins: conditioning, not an integer op)')
				elif verbose:
					print(line + '  OK')
		if was_training:
			self.nnet.train()
		for w in warnings_:
			if verbose:
				print(w)
		#if warnings_ and not verbose:
		#	print(f'[onnx-parity] note: {len(warnings_)} synthetic case(s) over tolerance on both signed and abs '
		#	      f'twins (float32 conditioning on out-of-distribution inputs); not raised. '
		#	      f'ONNX_PARITY_VERBOSE=1 for details.')

		if not failures:
			return True
		msg = ('[FATAL] torch/ONNX parity FAILED with a sign bias: the trained function is not the played '
		       'function. Known cause: integer floor division exported as Cast(float)->Div->Cast(int), which '
		       'TRUNCATES toward zero instead of flooring, so bits of negative values are wrong. '
		       'Fix: make the operand non-negative before dividing -- (v %% 256) // 2**i -- or use '
		       'torch.div(a, b, rounding_mode="floor"). SKIP_ONNX_PARITY=1 bypasses this check.')
		if raise_on_fail:
			raise RuntimeError(msg)
		print(msg)
		return False


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
		args.nn_version = checkpoint.get('nn_version', checkpoint['full_model'].version)

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
		if k not in ['state_dict', 'full_model', 'optim_state']:
			print(f'  {k}: {checkpoint[k]}')
	print(f'Board shape: {list(dummy_board.shape)}, valids shape: {list(dummy_valid_actions.shape)}')
