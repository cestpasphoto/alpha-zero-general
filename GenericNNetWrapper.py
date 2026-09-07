import os
import sys
import time
import pickle
import zlib

os.environ["OMP_NUM_THREADS"] = "1" # PyTorch more efficient this way

import numpy as np
from tqdm import tqdm
from time import sleep

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
		self.game = game   # kept so a refused tolerant load can be rolled back cleanly
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

	def train(self, examples, validation_set=None, save_folder=None, every=0):
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

				if validation_set and ((i_batch + batch_count*epoch) % every == 0):
					print(self.evaluate(validation_set))
					self.nnet.train()
					if (i_batch > 0) and save_folder:
						self.save_checkpoint(save_folder, filename=f'intermediary_{i_batch}.pt')

		t.close()
		
	def predict(self, board, valid_actions):
		"""
		board: np array with board
		"""
		# timing

		# preparing input
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
		
		# Legacy compatibility 
		if len(batch_info) == 5:
			i_thread, i_result, shared_memory, locks, network_id = batch_info
		else:
			i_thread, i_result, shared_memory, locks = batch_info
			network_id = 0

		# Store inputs in shared memory
		shared_memory[i_thread] = (
			np.expand_dims(board.astype(np.float32), 0),
			np.expand_dims(np.array(valid_actions).astype(np.bool_), 0),
			network_id
		)
		# Unblock next thread (= next MCTS or server), and wait for our turn
		locks[i_thread+1].release()
		locks[i_thread].acquire()

		# Retrieve results in shared memory
		ort_outs = shared_memory[i_result]
		pi, v = np.exp(ort_outs[0]), ort_outs[1]

		return pi, v

	def predict_server(self, nb_threads, shared_memory, locks, pnet=None):
		self.switch_target('inference')
		if pnet is not None:
			pnet.switch_target('inference')
		locks[0].release()

		while shared_memory[-1] <= 1:
			locks[-1].acquire() # Wait for all inputs

			reqs_0 = [i for i in range(nb_threads) if len(shared_memory[i]) > 2 and shared_memory[i][2] == 0]
			reqs_1 = [i for i in range(nb_threads) if len(shared_memory[i]) > 2 and shared_memory[i][2] == 1]
			
			# Fallback if network_id not provided
			if not reqs_0 and not reqs_1:
				reqs_0 = list(range(nb_threads))

			# inference batch for current network (nnet)
			if reqs_0:
				ort_outs_0 = self.ort_session.run(None, {
					'board'        : np.concatenate([shared_memory[i][0] for i in reqs_0]),
					'valid_actions': np.concatenate([shared_memory[i][1] for i in reqs_0]),
				})
				for j, idx in enumerate(reqs_0):
					shared_memory[idx+nb_threads] = (ort_outs_0[0][j], ort_outs_0[1][j])

			# inference batch for the sparring partner (pnet)
			if reqs_1 and pnet is not None and pnet.ort_session is not None:
				ort_outs_1 = pnet.ort_session.run(None, {
					'board'        : np.concatenate([shared_memory[i][0] for i in reqs_1]),
					'valid_actions': np.concatenate([shared_memory[i][1] for i in reqs_1]),
				})
				for j, idx in enumerate(reqs_1):
					shared_memory[idx+nb_threads] = (ort_outs_1[0][j], ort_outs_1[1][j])

			locks[0].release() # Unblock 1st thread

	def evaluate(self, validation_set):
		# print()
		# print(f'LR = {scheduler.get_last_lr()[0]:.1e}', end=' ')

		# Evaluation
		self.nnet.eval()
		with torch.no_grad():
			picked_examples = [pickle.loads(zlib.decompress(e)) for e in validation_set]
			boards, pis, vs, valid_actions, qs = list(zip(*picked_examples))
			boards = torch.FloatTensor(np.array(boards).astype(np.float32))
			valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_))
			target_pis = torch.FloatTensor(np.array(pis).astype(np.float32))
			target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))
			target_qs = torch.FloatTensor(np.array(qs).astype(np.float32))

			# compute output
			out_pi, out_v = self.nnet(boards, valid_actions)
			total_loss = self.loss_pi(target_pis, out_pi) + self.loss_v(target_vs, target_qs, out_v)
			return total_loss.item()

	def loss_pi(self, targets, outputs):
		loss_ = torch.nn.KLDivLoss(reduction="batchmean")
		return loss_(outputs, targets)

		# loss_ = torch.nn.CrossEntropyLoss()
		# return loss_(outputs, targets)

		# return -torch.sum(torch.log(targets) * torch.exp(outputs)) / targets.size()[0]

	def loss_v(self, targets_V, targets_Q, outputs):
		targets = (targets_V + self.args['q_weight'] * targets_Q) / (1+self.args['q_weight'])
		return torch.sum((targets - outputs) ** 2) / (targets_V.size()[0] * targets_V.size()[-1]) # Normalize by batch size * nb of players

	def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', additional_keys={}):
		filepath = os.path.join(folder, filename)
		if not os.path.exists(folder):
			# print("Checkpoint Directory does not exist! Making directory {}".format(folder))
			os.mkdir(folder)
		# else:
		#     print("Checkpoint Directory exists! ")

		data = {
			'state_dict': self.nnet.state_dict(),
			'full_model': self.nnet,
			# Explicit version key: version check no longer relies solely on the
			# pickled full_model object (which can be corrupted by cross-arch transfer).
			'nn_version': self.nnet.version,
			# Inference-semantics flag, stored PER CHECKPOINT so that two copies
			# of the same weights can be evaluated against each other.
			'unsigned_bits': getattr(self.nnet, 'unsigned_bits', False),
		}
		data.update(additional_keys)
		torch.save(data, filepath)

	def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
		# https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
		filepath = os.path.join(folder, filename)
		if not os.path.exists(filepath):
			print("No model in path {}".format(filepath))
			return			
		try:
			checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
			self.load_network(checkpoint, strict=(self.args['nn_version']>0))
		except:
			print("MODEL {} CAN'T BE READ but file exists".format(filepath))
			return
		self.switch_target('just_loaded')
		return checkpoint
			
	def load_network(self, checkpoint, strict=False):
		def load_not_strict(network_state_to_load, target_network):
			target_state = target_network.state_dict()
			for name, params in network_state_to_load.items():
				if name in target_state:
					target_params = target_state[name]
					if target_params.shape == params.shape:
						target_params.copy_(params)
						# print(f'no problem to copy {name}')
					elif target_params.dim() == params.dim():
						if len(target_params.shape) == 1:
							min_size = min(target_params.shape[0], params.shape[0])
							target_params[:min_size] = params[:min_size]
						elif len(target_params.shape) == 2:
							min_size_0, min_size_1 = min(target_params.shape[0], params.shape[0]), min(target_params.shape[1], params.shape[1])
							target_params[:min_size_0, :min_size_1] = params[:min_size_0, :min_size_1]
						elif len(target_params.shape) == 3:
							min_size_0, min_size_1, min_size_2 = min(target_params.shape[0], params.shape[0]), min(target_params.shape[1], params.shape[1]), min(target_params.shape[2], params.shape[2])
							target_params[:min_size_0, :min_size_1, :min_size_2] = params[:min_size_0, :min_size_1, :min_size_2]
						elif len(target_params.shape) == 4:
							min_size_0, min_size_1, min_size_2, min_size_3 = min(target_params.shape[0], params.shape[0]), min(target_params.shape[1], params.shape[1]), min(target_params.shape[2], params.shape[2]), min(target_params.shape[3], params.shape[3])
							target_params[:min_size_0, :min_size_1, :min_size_2, :min_size_3] = params[:min_size_0, :min_size_1, :min_size_2, :min_size_3]
						else:
							raise Exception('Unsupported number of dimensions')

						print(f'{name}: load {params.shape}  target {target_params.shape}, used {(min_size_0)}')
					else:
						print(f'{name}: couldnt match loaded {params.shape}  and target {target_params.shape}, using standard initialization')
						
				# else:
				# 	print(f'hasnt loaded layer {name} because not in target')

		# Prefer the explicit 'nn_version' key (written since the fix); fall back to
		# full_model.version for legacy checkpoints that pre-date this key.
		ckpt_version = checkpoint.get('nn_version', checkpoint['full_model'].version)
		if 'unsigned_bits' in checkpoint and hasattr(self.nnet, 'stem'):
		    self.nnet.unsigned_bits = bool(checkpoint['unsigned_bits'])
		    self.nnet.stem.unsigned_bits = self.nnet.unsigned_bits
		    # Keep args in sync: Coach rebuilds the competitor net from nnet.args
		    # (Coach.py l.30), so an attribute-only update leaves pnet on the old
		    # semantics until its first load_checkpoint().
		    self.nnet.args['unsigned_bits'] = self.nnet.unsigned_bits

		if strict and (ckpt_version != self.args['nn_version']):
			print('Checkpoint includes NN version', ckpt_version, ', but you ask version', self.args['nn_version'], ' so not loading it and initiate knowledge transfer')
			self.requestKnowledgeTransfer = True
			return

		try:
			self.nnet.load_state_dict(checkpoint['state_dict'])
			self.nnet.version = ckpt_version
		except:
			# Same architecture version, but checkpoint written BEFORE some purely
			# ADDITIVE, zero-initialised tensors existed. Accept it if and only if
			# nothing else differs -- the loaded net then computes exactly the
			# function the checkpoint encoded. Any other case falls through to the
			# pre-existing behaviour below.
			if self._load_additive_compatible(checkpoint['state_dict'], ckpt_version):
				return
			if strict:
				print('Cant load NN', ckpt_version, 'in checkpoint, so initiate knowledge transfer')
				self.requestKnowledgeTransfer = True
			else:
				if self.nnet.version > 0:
					try:
						load_not_strict(checkpoint['state_dict'], self.nnet)
						print('Could load state dict but NOT STRICT, saved archi-version was', ckpt_version)
					except:
						# GUARD: only replace self.nnet with the full pickled model if the
						# versions match (same class). A cross-version replacement (e.g.
						# V62 SmallworldNNet replacing a V72 SmallworldGraphNNet) corrupts
						# all subsequent saves: full_model.version becomes the OLD version
						# and the next strict load wrongly fires the version-mismatch path.
						if ckpt_version == self.nnet.version:
							self.nnet = self._adopt_full_model(checkpoint['full_model'])
							print('Had to load full model AS IS (V%s), WONT BE UPDATED' % ckpt_version)
							if input("Continue? [y|n]") != "y":
								sys.exit()
						else:
							print('load_not_strict failed V%s->V%s; keeping random init for V%s' % (
								ckpt_version, self.nnet.version, self.nnet.version))
				else:
					# nn_version=-1 (e.g. GenericNNetWrapper.py -i <file> without -V):
					# accept the full pickled model as-is (diagnostic / standalone use).
					self.nnet = self._adopt_full_model(checkpoint['full_model'])


	def _adopt_full_model(self, model):
		# A pickled model carries the __dict__ it had when it was SAVED, but its
		# methods come from the CURRENT class. If the game's net grew submodules
		# since, ask it to repair itself before we use it.
		if hasattr(model, 'upgrade_legacy'):
			model.upgrade_legacy()
		return model

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
		print(f'Loaded V{ckpt_version} checkpoint; {len(result.missing_keys)} additive tensors kept at zero-init:', result.missing_keys)
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
		opts.intra_op_num_threads, opts.inter_op_num_threads, opts.inter_op_num_threads = 1, 1, ort.ExecutionMode.ORT_SEQUENTIAL
		self.ort_session = ort.InferenceSession(temporary_file, sess_options=opts, providers=['CPUExecutionProvider'])
		os.remove(temporary_file)
		# GUARDRAIL: the function that PLAYS must be the function that was TRAINED.
		# Single choke point: every ONNX session (self-play, arena, daemon, pit) is born here.
		if os.environ.get('SKIP_ONNX_PARITY') != '1':
			self._assert_onnx_parity(verbose=(os.environ.get('ONNX_PARITY_VERBOSE') == '1'))


	def _assert_onnx_parity(self, n_synth=256, batch_size=8, tol_pi=1e-4, tol_v=1e-5,
	                        seed=0, verbose=False, raise_on_fail=True):
		"""
		Compare the torch module and the freshly exported ONNX session on the SAME
		inputs. Raises RuntimeError on mismatch. Cost ~0.2 s per export.

		Why synthetic boards over the full int8 range rather than real positions:
		the known failure mode (integer floor-div lowered to a truncating cast in
		ONNX) only shows on NEGATIVE values, and a sample of "realistic" boards may
		not exercise the column that carries them. Random boards are not legal game
		states -- that is fine, we are comparing two implementations of the same
		function, not playing.

		Batch 8 and batch 1 are both tested: the graph is exported at batch 1 with
		dynamic axes, but predict_server() runs it batched.

		Tolerances: on a healthy net the measured gap is ~4e-6 on log-probs and
		~5e-7 on v, so 1e-4 / 1e-5 leave >20x of headroom, while a single wrong bit
		moves logits by ~4e-2.
		"""
		import numpy as _np
		rng = _np.random.default_rng(seed)
		boards = rng.integers(-128, 128, size=(n_synth,) + tuple(self.board_size)).astype(_np.float32)
		boards[:n_synth // 2] = _np.abs(boards[:n_synth // 2])   # half with NO negative value,
		valids = rng.random((n_synth, self.action_size)) > 0.5    # so the diagnostic below can
		valids[:, 0] = True                      # never an all-illegal row; separate the two groups

		was_training = self.nnet.training
		self.nnet.eval()
		with torch.no_grad():
			pi_t, v_t = self.nnet(torch.from_numpy(boards), torch.from_numpy(valids))
		pi_t, v_t = pi_t.numpy(), v_t.numpy()
		if was_training:
			self.nnet.train()

		failures = []
		for bs in (batch_size, 1):
			pi_o, v_o = _np.empty_like(pi_t), _np.empty_like(v_t)
			for s in range(0, n_synth, bs):
				e = min(s + bs, n_synth)
				out = self.ort_session.run(None, {'board': boards[s:e], 'valid_actions': valids[s:e]})
				pi_o[s:e], v_o[s:e] = out[0], out[1]
			dpi = float(_np.abs(pi_t - pi_o)[valids].max())      # legal actions only
			dv = float(_np.abs(v_t - v_o).max())
			ok = (dpi <= tol_pi) and (dv <= tol_v)
			if verbose or not ok:
				print(f'[onnx-parity] n={n_synth} bs={bs}  max|dpi|={dpi:.3e} (tol {tol_pi:.0e})  '
				      f'max|dv|={dv:.3e} (tol {tol_v:.0e})  {"OK" if ok else "*** MISMATCH ***"}')
			if not ok:
				over = _np.where(valids, _np.abs(pi_t - pi_o), 0.).max(axis=1) > tol_pi
				has_neg = (boards < 0).any(axis=(1, 2))
				print(f'[onnx-parity] {over.mean():.0%} of boards over tolerance; among them '
				      f'{has_neg[over].mean():.0%} contain a negative value (vs {has_neg.mean():.0%} '
				      f'overall). A strong bias toward negatives points at an integer op.')
				failures.append(bs)

		if not failures:
			return True
		msg = ('[FATAL] torch/ONNX parity FAILED: the trained function is not the played function. '
		       'Known cause: integer floor division exported as Cast(float)->Div->Cast(int), which '
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
	import argparse
	import os.path
	import time
	from GameSwitcher import import_game

	parser = argparse.ArgumentParser(description='NNet loader')
	parser.add_argument('game'               , action='store', default='splendor', help='The name of the game to play')
	parser.add_argument('--input'      , '-i', action='store', default=None , help='Input NN to load')
	parser.add_argument('--output'     , '-o', action='store', default=None , help='Prefix for output NN')
	parser.add_argument('--training'   , '-T', action='store', default=None , help='')
	parser.add_argument('--test'       , '-t', action='store', default=None , help='')

	parser.add_argument('--learn-rate' , '-l' , action='store', default=0.0003, type=float, help='')
	parser.add_argument('--dropout'    , '-d' , action='store', default=0.3   , type=float, help='')
	parser.add_argument('--epochs'     , '-p' , action='store', default=2    , type=int  , help='')
	parser.add_argument('--batch-size' , '-b' , action='store', default=32   , type=int  , help='')
	parser.add_argument('--nb-samples' , '-N' , action='store', default=9999 , type=int  , help='How many samples (in thousands)')
	parser.add_argument('--nn-version' , '-V' , action='store', default=-1   , type=int  , help='Which architecture to choose')
	parser.add_argument('--q-weight'   , '-q' , action='store', default=0.5  , type=float, help='Weight for mixing Q into value loss')
	args = parser.parse_args()
	Game, NNet, players, NUMBER_PLAYERS = import_game(args.game)

	output = (args.output if args.output else 'output_') + str(int(time.time()))[-6:]

	g = Game()
	nn_args = dict(
		lr=args.learn_rate,
		dropout=args.dropout,
		epochs=args.epochs,
		batch_size=args.batch_size,
		nn_version=args.nn_version,
		learn_rate=args.learn_rate,
		no_compression=False,
		q_weight=args.q_weight,
	)
	nnet = NNet(g, nn_args)
	if args.input:
		nnet.load_checkpoint(os.path.dirname(args.input), os.path.basename(args.input))
	elif args.nn_version == -1:
		raise Exception("You have to specify at least a NN file to load or a NN version")

	from fvcore.nn import FlopCountAnalysis
	dummy_board         = torch.randn(g.getBoardSize(), dtype=torch.float32).unsqueeze(0)
	dummy_valid_actions = torch.BoolTensor(torch.randn(1, g.getActionSize())>0.5)
	nnet.nnet.eval()
	flops = FlopCountAnalysis(nnet.nnet, (dummy_board, dummy_valid_actions))
	flops.unsupported_ops_warnings(False)
	# flops.uncalled_modules_warnings(False)
	print(f'V{nnet.nnet.version} -> {flops.total()/1000000:.1f} MFlops, nb params {nnet.number_params()[0]:.2e}')
	# print(flops.by_module().most_common(15))
	# print({ k:v//1000 for k,v in flops.by_module().items() if k.count('.') <= 2 })
	# breakpoint()

	if not args.training:
		if args.input:
			checkpoint = torch.load(args.input, map_location='cpu', weights_only=False)
			for k in sorted(checkpoint.keys()):
				if k not in ['state_dict', 'full_model', 'optim_state']:
					print(f'  {k}: {checkpoint[k]}')
			print(f'Board shape: {list(dummy_board.shape)}, valids shape: {list(dummy_valid_actions.shape)}')
		exit()
	with open(args.training, "rb") as f:
		examples = pickle.load(f)
	trainExamples = []
	for e in examples:
		trainExamples.extend(e)
	if args.test is None:
		splitNumber = len(trainExamples) // 10
		testExamples, trainExamples = trainExamples[-splitNumber:], trainExamples[:-splitNumber]
	else:
		with open(args.test, "rb") as f:
			examples = pickle.load(f)
		testExamples = []
		for e in examples:
			testExamples.extend(e)
	trainExamples = trainExamples[-args.nb_samples*1000:]
	print(f'Number of samples: training {len(trainExamples)}, testing {len(testExamples)}; number of epochs {args.epochs}')

	# print({ k:v//1000 for k,v in flops.by_module().items() if k.count('.') <= 1 })
	# breakpoint()
	
	# trainExamples_small = trainExamples[::30]
	# testExamples_small = testExamples[::30]
	# nnet.args['learn_rate'], nnet.args['lr'], nnet.args['batch_size'] = 3e-2, 3e-2, 32
	# nnet.optimizer = None
	# save_every = 1e5 // nnet.args['batch_size']
	# nnet.train(trainExamples_small, testExamples_small, '', save_every)

	# nnet.args['learn_rate'], nnet.args['lr'], nnet.args['batch_size'] = 3e-4, 3e-4, 512
	# nnet.optimizer = None
	save_every = (1e5 // nnet.args['batch_size']) - 1
	nnet.train(trainExamples, testExamples, output, save_every)

	nnet.save_checkpoint(output, filename='last.pt')
