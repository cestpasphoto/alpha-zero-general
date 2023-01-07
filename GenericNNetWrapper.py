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
torch.set_num_threads(1) # PyTorch more efficient this way

class GenericNNetWrapper(NeuralNet):
	def __init__(self, game, nn_args):
		self.args = nn_args
		self.device = {
			'training' : 'cpu', #'cuda' if torch.cuda.is_available() else 'cpu',
			'inference': 'onnx',
			'just_loaded': 'cpu',
		}
		self.current_mode = 'cpu'
		self.init_nnet(game, nn_args)
		self.ort_session = None

		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.max_diff = game.getMaxScoreDiff()
		self.num_players = game.num_players
		self.optimizer = None
		self.requestKnowledgeTransfer = False

	def init_nnet(self, game, nn_args):
		pass

	def train(self, examples, validation_set=None, save_folder=None, every=0):
		"""
		examples: list of examples, each example is of form (board, pi, v)
		"""
		self.switch_target('training')

		if self.optimizer is None:
			self.optimizer = optim.AdamW(self.nnet.parameters(), lr=self.args['learn_rate'])
		batch_count = int(len(examples) / self.args['batch_size'])

		if True:
			scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args['learn_rate'], steps_per_epoch=batch_count, epochs=self.args['epochs'])
		else:
			batch_count = batch_count // 5
			every = every // 5
			validation_set = validation_set[::5]
			scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-3, total_iters=batch_count)

		examples_weights = self.compute_surprise_weights(examples) if self.args['surprise_weight'] else None

		t = tqdm(total=self.args['epochs'] * batch_count, desc='Train ep0', colour='blue', ncols=120, mininterval=0.5, disable=None)
		for epoch in range(self.args['epochs']):
			t.set_description(f'Train ep{epoch + 1}')
			self.nnet.train()
			pi_losses, v_losses, scdiff_losses = AverageMeter(), AverageMeter(), AverageMeter()
	
			for i_batch in range(batch_count):
				sample_ids = np.random.choice(len(examples), size=self.args['batch_size'], replace=False, p=examples_weights)
				boards, pis, vs, scdiffs, valid_actions, surprises = self.pick_examples(examples, sample_ids)
				boards = torch.FloatTensor(self.reshape_boards(np.array(boards)).astype(np.float32))
				valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_))
				target_pis = torch.FloatTensor(np.array(pis).astype(np.float32))
				target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))
				target_scdiffs = torch.FloatTensor(np.zeros((len(scdiffs), 2*self.max_diff+1, self.num_players)).astype(np.float32))
				for i in range(len(scdiffs)):
					score_diff = (scdiffs[i] + self.max_diff).clip(0, 2*self.max_diff)
					for player in range(self.num_players):
						target_scdiffs[i, score_diff[player], player] = 1

				# predict
				self.optimizer.zero_grad(set_to_none=True)
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
				total_loss.backward()
				self.optimizer.step()
				scheduler.step()

				t.update()

				if validation_set and ((i_batch + batch_count*epoch) % every == 0):
					# print()
					# print(f'LR = {scheduler.get_last_lr()[0]:.1e}', end=' ')
					# Evaluation
					self.nnet.eval()
					with torch.no_grad():
						picked_examples = [pickle.loads(zlib.decompress(e)) for e in validation_set]
						boards, pis, vs, scdiffs, valid_actions, surprises = list(zip(*picked_examples))
						boards = torch.FloatTensor(self.reshape_boards(np.array(boards)).astype(np.float32))
						valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_))
						target_pis = torch.FloatTensor(np.array(pis).astype(np.float32))
						target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))
						target_scdiffs = torch.FloatTensor(np.zeros((len(scdiffs), 2*self.max_diff+1, self.num_players)).astype(np.float32))
						for i in range(len(scdiffs)):
							score_diff = (scdiffs[i] + self.max_diff).clip(0, 2*self.max_diff)
							for player in range(self.num_players):
								target_scdiffs[i, score_diff[player], player] = 1

						# compute output
						out_pi, out_v, out_scdiff = self.nnet(boards, valid_actions)
						l_pi = self.loss_pi(target_pis, out_pi)
						l_v = self.loss_v(target_vs, out_v)
						l_scdiff_c = self.loss_scdiff_cdf(target_scdiffs, out_scdiff)
						l_scdiff_p = self.loss_scdiff_pdf(target_scdiffs, out_scdiff)
						total_loss = l_pi + self.args['vl_weight']*l_v + l_scdiff_c + l_scdiff_p
						test_loss = total_loss.item()
						print(test_loss)
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
				'board': board.astype(np.float32).reshape((-1, self.nb_vect, self.vect_dim)),
				'valid_actions': np.array(valid_actions).astype(np.bool_).reshape((-1, self.action_size)),
			})
			pi, v = np.exp(ort_outs[0][0]), ort_outs[1][0]
			return pi, v

		else:
			board = torch.FloatTensor(board.astype(np.float32)).reshape((-1, self.nb_vect, self.vect_dim))
			valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_)).reshape((-1, self.action_size))
			if self.current_mode == 'cuda':
				board, valid_actions = board.contiguous().cuda(), valid_actions.contiguous().cuda()
			self.nnet.eval()
			with torch.no_grad():
				pi, v, _ = self.nnet(board, valid_actions)
			pi, v = torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
			return pi, v

	def loss_pi(self, targets, outputs):
		return -torch.sum(targets * outputs) / targets.size()[0]

	def loss_v(self, targets, outputs):
		# targets = (targets_V + self.args['q_weight'] * targets_Q) / (1+self.args['q_weight'])
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
						params.copy_(target_params)
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

		if strict and (checkpoint['full_model'].version != self.args['nn_version']):
			print('Checkpoint includes NN version', checkpoint['full_model'].version, ', but you ask version', self.args['nn_version'], ' so not loading it and initiate knowledge transfer')
			self.requestKnowledgeTransfer = True
			return

		try:
			self.nnet.load_state_dict(checkpoint['state_dict'])
		except:
			if strict:
				print('Cant load NN ', checkpoint['full_model'].version, 'in checkpoint, so initiate knowledge transfer')
				self.requestKnowledgeTransfer = True
			else:
				if self.nnet.version > 0:
					try:
						load_not_strict(checkpoint['state_dict'], self.nnet)
						print('Could load state dict but NOT STRICT, saved archi-version was', checkpoint['full_model'].version)
					except:
						self.nnet = checkpoint['full_model']
						print('Had to load full model AS IS, saved archi-version was', checkpoint['full_model'].version, 'and WONT BE UPDATED')
						if input("Continue? [y|n]") != "y":
							sys.exit()
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

		# 		Measured avg duration of inference over 5 tests of 20k calls, 1 inference each time
		#       Checked that variability is < ±0.05, W means warning emitted
		#
		#           ort_protocol:     9       10      11      12      13      14       15       16
		# date     torch_v  ort_v
		# Apr 21    1.8.0   1.7.0     2.14W   2.20W   2.21    2.22    2.26
		# Sep 21    1.9.1   1.8.1     2.20W   2.23W   2.25    2.26    2.27
		# Oct 21    1.10.1  1.9.0     2.29W   2.31W   2.34    2.34    2.38    2.38W
		# Dec 21    1.10.2  1.10.0    2.12W   2.17W   2.20    2.20    2.23    2.21W
		# Apr 22    1.11    1.11.1    1.95W   1.97W   2.00    2.01    2.04    2.04W    2.04W
		# Aug 22    1.12.1  1.12.1    2.03W   2.03W   2.07    2.06    2.10    2.09     2.09     2.09
		#
		# Conclusion is that is that (slightly) faster protocol is 11

		torch.onnx.export(
			self.nnet,
			(dummy_board, dummy_valid_actions),
			temporary_file,
			opset_version=11, 	# best ONNX protocol, see comment above
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
	from santorini.SantoriniGame import SantoriniGame as Game
	from santorini.NNet import NNetWrapper as nn
	torch.set_num_threads(2) # PyTorch more efficient this way

	parser = argparse.ArgumentParser(description='NNet loader')
	parser.add_argument('--input'      , '-i', action='store', default=None , help='Input NN to load')
	parser.add_argument('--output'     , '-o', action='store', default=None , help='Prefix for output NN')
	parser.add_argument('--training'   , '-T', action='store', default='../results/new_training.examples' , help='')
	parser.add_argument('--test'       , '-t', action='store', default='../results/new_testing.examples'  , help='')

	parser.add_argument('--learn-rate' , '-l' , action='store', default=0.0003, type=float, help='')
	parser.add_argument('--dropout'    , '-d' , action='store', default=0.    , type=float, help='')
	parser.add_argument('--epochs'     , '-p' , action='store', default=2    , type=int  , help='')
	parser.add_argument('--batch-size' , '-b' , action='store', default=256  , type=int  , help='')
	parser.add_argument('--nb-samples' , '-N' , action='store', default=9999 , type=int  , help='How many samples (in thousands)')
	parser.add_argument('--nn-version' , '-V' , action='store', default=24   , type=int  , help='Which architecture to choose')
	parser.add_argument('--vl-weight'  , '-v' , action='store', default=4.   , type=float, help='Weight for value loss')
	parser.add_argument('--details'    , '-D' , action='store', default=[128,0,6], type=int, nargs=3, help='Details for NN 80')
	args = parser.parse_args()	

	output = (args.output if args.output else 'output_') + str(int(time.time()))[-6:]

	g = Game()
	nn_args = dict(
		lr=args.learn_rate,
		dropout=args.dropout,
		epochs=args.epochs,
		batch_size=args.batch_size,
		nn_version=args.nn_version,
		learn_rate=args.learn_rate,
		vl_weight=args.vl_weight,
		surprise_weight=False,
		no_compression=False,
		n_filters=args.details[0],
		expansion=['small', 'constant', 'progressive'][args.details[1]],
		depth=args.details[2],
	)
	nnet = nn(g, nn_args)
	if args.input:
		nnet.load_checkpoint(os.path.dirname(args.input), os.path.basename(args.input))
	
	from fvcore.nn import FlopCountAnalysis
	dummy_board         = torch.randn(1, 25, 3, dtype=torch.float32)
	dummy_valid_actions = torch.BoolTensor(torch.randn(1, 162)>0.5)
	flops = FlopCountAnalysis(nnet.nnet, (dummy_board, dummy_valid_actions))
	flops.unsupported_ops_warnings(False)
	flops.uncalled_modules_warnings(False)
	print(f'V{args.nn_version} {args.details} -> {flops.total()//1000000} MFlops, nb params {nnet.number_params()[0]:.2e}')
	if not (2 <= flops.total()//1000000 <= 20):
		exit()

	with open(args.training, "rb") as f:
		examples = pickle.load(f)
	trainExamples = []
	for e in examples:
		trainExamples.extend(e)
	trainExamples = trainExamples[-args.nb_samples*1000:]
	with open(args.test, "rb") as f:
		examples = pickle.load(f)
	testExamples = []
	for e in examples:
		testExamples.extend(e)
	print(f'Number of samples: training {len(trainExamples)}, testing {len(testExamples)}; number of epochs {args.epochs}')

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
