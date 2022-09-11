import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1" # PyTorch more efficient this way

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

# import warnings
# warnings.filterwarnings('ignore', category=UserWarning)
import torch
import torch.optim as optim
import torch.onnx
import onnxruntime as ort
torch.set_num_threads(1) # PyTorch more efficient this way

from .TLPNNet import TLPNNet as mnnet

class NNetWrapper(NeuralNet):
	def __init__(self, game, nn_args):
		self.args = nn_args
		self.device = {
			'training' : 'cpu', #'cuda' if torch.cuda.is_available() else 'cpu',
			'inference': 'onnx',
			'just_loaded': 'cpu',
		}
		self.current_mode = 'cpu'
		self.nnet = mnnet(game, nn_args)
		self.ort_session = None

		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.max_diff = game.getMaxScoreDiff()
		self.num_players = game.num_players
		self.optimizer = None

	def train(self, examples):
		"""
		examples: list of examples, each example is of form (board, pi, v)
		"""
		self.switch_target('training')

		if self.optimizer is None:
			self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.args['learn_rate'])		
		batch_count = int(len(examples) / self.args['batch_size'])

		if self.args['cyclic_lr']:
			scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args['learn_rate']*10, anneal_strategy='cos', total_steps=self.args['epochs']*batch_count)
			# scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args['learn_rate']*10, anneal_strategy='linear', total_steps=self.args['epochs']*batch_count)

		examples_weights = self.compute_surprise_weights(examples) if self.args['surprise_weight'] else None

		t = tqdm(total=self.args['epochs'] * batch_count, desc='Train ep0', colour='blue', ncols=120, mininterval=0.5, disable=None)
		for epoch in range(self.args['epochs']):
			t.set_description(f'Train ep{epoch + 1}')
			self.nnet.train()
			pi_losses, v_losses, scdiff_losses = AverageMeter(), AverageMeter(), AverageMeter()
	
			for _ in range(batch_count):
				sample_ids = np.random.choice(len(examples), size=self.args['batch_size'], replace=False, p=examples_weights)
				boards, pis, vs, scdiffs, valid_actions, surprises = self.pick_examples(examples, sample_ids)
				boards = torch.FloatTensor(np.array(boards).astype(np.float32))
				valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_))
				target_pis = torch.FloatTensor(np.array(pis).astype(np.float32))
				target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))
				target_scdiffs = torch.FloatTensor(np.zeros((len(scdiffs), 2*self.max_diff+1, self.num_players)).astype(np.float32))
				for i in range(len(scdiffs)):
					score_diff = (scdiffs[i] + self.max_diff).clip(0, 2*self.max_diff)
					for player in range(self.num_players):
						target_scdiffs[i, score_diff[player], player] = 1

				# predict
				if self.device['training'] == 'cuda':
					boards, target_pis, target_vs, valid_actions, target_scdiffs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda(), valid_actions.contiguous().cuda(), target_scdiffs.contiguous().cuda()

				# compute output
				out_pi, out_v, out_scdiff = self.nnet(boards, valid_actions)
				l_pi = self.loss_pi(target_pis, out_pi)
				l_v = self.loss_v(target_vs, out_v)
				l_scdiff_c = self.loss_scdiff_cdf(target_scdiffs, out_scdiff)
				l_scdiff_p = self.loss_scdiff_pdf(target_scdiffs, out_scdiff)
				total_loss = l_pi + self.args['vl_weight']*l_v + l_scdiff_c + l_scdiff_p

				# record loss
				pi_losses.update(l_pi.item(), boards.size(0))
				v_losses.update(l_v.item(), boards.size(0))
				scdiff_losses.update(l_scdiff_c.item() + l_scdiff_p.item(), boards.size(0))
				t.set_postfix(PI=pi_losses, V=v_losses, SD=scdiff_losses, refresh=False)

				# compute gradient and do SGD step
				self.optimizer.zero_grad(set_to_none=True)
				total_loss.backward()
				self.optimizer.step()
				if self.args['cyclic_lr']:
					scheduler.step()

				t.update()
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
				'board': board.astype(np.float32).reshape((-1, self.nb_vect, self.vect_dim)),
				'valid_actions': np.array(valid_actions).astype(np.bool_).reshape((-1, self.action_size)),
			})
			pi, v = np.exp(ort_outs[0][0]), ort_outs[1][0]
			return pi, v

		else:
			board = torch.FloatTensor(board.astype(np.float32))
			valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_))
			if self.current_mode == 'cuda':
				board, valid_actions = board.contiguous().cuda(), valid_actions.contiguous().cuda()
			self.nnet.eval()
			with torch.no_grad():
				pi, v, _ = self.nnet(board, valid_actions)
			pi, v = torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0][0]
			return pi, v

	def loss_pi(self, targets, outputs):
		return -torch.sum(targets * outputs) / targets.size()[0]

	def loss_v(self, targets, outputs):
		return torch.sum((targets - outputs) ** 2) / (targets.size()[0] * targets.size()[-1]) # Normalize by batch size * nb of players

	def loss_scdiff_cdf(self, targets, outputs):
		l2_diff = torch.square(torch.cumsum(targets, axis=1) - torch.cumsum(torch.exp(outputs), axis=1))
		return 0.02 * torch.sum(l2_diff) / (targets.size()[0] * targets.size()[-1]) # Normalize by batch size * nb of scdiffs

	def loss_scdiff_pdf(self, targets, outputs):
		cross_entropy = -torch.sum( torch.mul(targets, outputs) )
		return 0.02 * cross_entropy / (targets.size()[0] * targets.size()[-1]) # Normalize by batch size * nb of scdiffs

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
			checkpoint = torch.load(filepath, map_location='cpu')
			self.load_network(checkpoint, strict=False)
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
						params.copy_(target_params)
						# print(f'no problem to copy {name}')
					else:
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
				# else:
				# 	print(f'hasnt loaded layer {name} because not in target')
				
		try:
			self.nnet.load_state_dict(checkpoint['state_dict'])
		except:
			if self.nnet.version > 0:
				try:
					load_not_strict(checkpoint['state_dict'], self.nnet)
					print('Could load state dict but NOT STRICT, saved archi-version was', checkpoint['full_model'].version)
				except:
					self.nnet = checkpoint['full_model']
					print('Had to load full model AS IS, saved archi-version was', checkpoint['full_model'].version, 'and wont be updated')
			else:
				self.nnet = checkpoint['full_model']


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
		dummy_board         = torch.randn(1, self.nb_vect, self.vect_dim, dtype=torch.float32)
		dummy_valid_actions = torch.BoolTensor(torch.randn(1, self.action_size)>0.5)
		self.nnet.to('cpu')
		self.nnet.eval()

		temporary_file = 'nn_export_' + str( int(time.time()*1000)%1000000 ) + '.onnx'

		torch.onnx.export(
			self.nnet,
			(dummy_board, dummy_valid_actions),
			temporary_file,
			export_params=True,
			opset_version=12, # the ONNX version to export the model to
			do_constant_folding=True,
			input_names = ['board', 'valid_actions'],
			output_names = ['pi', 'v', 'scdiffs'],
			dynamic_axes={
				'board'        : {0: 'batch_size'},
				'valid_actions': {0: 'batch_size'},
				'pi'           : {0: 'batch_size'},
				'v'            : {0: 'batch_size'},
				'scdiffs'      : {0: 'batch_size'},
			}
		)

		opts = ort.SessionOptions()
		opts.intra_op_num_threads, opts.inter_op_num_threads, opts.inter_op_num_threads = 1, 1, ort.ExecutionMode.ORT_SEQUENTIAL
		self.ort_session = ort.InferenceSession(temporary_file, sess_options=opts)
		os.remove(temporary_file)

	def pick_examples(self, examples, sample_ids):
		if self.args['no_compression']:
			picked_examples = [examples[i] for i in sample_ids]
		else: 
			picked_examples = [pickle.loads(zlib.decompress(examples[i])) for i in sample_ids]
		return list(zip(*picked_examples))

	def compute_surprise_weights(self, examples):
		if self.args['no_compression']:
			examples_surprises = np.array([x[-1] for x in examples])
		else:
			examples_surprises = np.array([pickle.loads(zlib.decompress(x))[-1] for x in examples])
		examples_weights = examples_surprises / examples_surprises.sum() + 1./len(examples_surprises)
		examples_weights = examples_weights / examples_weights.sum()

		return examples_weights
