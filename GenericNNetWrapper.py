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
				boards = torch.FloatTensor(self.reshape_boards(np.array(boards)).astype(np.float32))
				valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_))
				target_pis = torch.FloatTensor(np.array(pis).astype(np.float32))
				target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))
				target_qs = torch.FloatTensor(np.array(qs).astype(np.float32))

				# predict
				optimizer.zero_grad(set_to_none=True)
				out_pi, out_v = self.nnet(boards, valid_actions)
				l_pi, l_v = self.loss_pi(target_pis, out_pi), self.loss_v(target_vs, target_qs, out_v)
				total_loss = l_pi + l_v

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
				pi, v = self.nnet(board, valid_actions)
			pi, v = torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
			return pi, v

	def predict_client(self, board, valid_actions, batch_info):
		if self.current_mode != 'onnx':
			raise Exception('Batch prediction only in ONNX mode')
		i_thread, i_result, shared_memory, locks = batch_info

		# Store inputs in shared memory
		shared_memory[i_thread] = (
			board.astype(np.float32).reshape((-1, self.nb_vect, self.vect_dim)),
			np.array(valid_actions).astype(np.bool_).reshape((-1, self.action_size)),
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

			# Batch inference
			ort_outs = self.ort_session.run(None, {
				'board'        : np.concatenate([x[0] for x in shared_memory[:nb_threads]]),
				'valid_actions': np.concatenate([x[1] for x in shared_memory[:nb_threads]]),
			})
			for i in range(nb_threads):
				shared_memory[i+nb_threads] = (ort_outs[0][i], ort_outs[1][i])

			locks[0].release() # Unblock 1st thread

	def evaluate(self, validation_set):
		# print()
		# print(f'LR = {scheduler.get_last_lr()[0]:.1e}', end=' ')

		# Evaluation
		self.nnet.eval()
		with torch.no_grad():
			picked_examples = [pickle.loads(zlib.decompress(e)) for e in validation_set]
			boards, pis, vs, valid_actions, qs = list(zip(*picked_examples))
			boards = torch.FloatTensor(self.reshape_boards(np.array(boards)).astype(np.float32))
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
			self.nnet.version = checkpoint['full_model'].version
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
			output_names = ['pi', 'v'],
			dynamic_axes={
				'board'        : {0: 'batch_size'},
				'valid_actions': {0: 'batch_size'},
				'pi'           : {0: 'batch_size'},
				'v'            : {0: 'batch_size'},
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
	from minivilles.MinivillesGame import MinivillesGame as Game
	from minivilles.NNet import NNetWrapper as nn

	parser = argparse.ArgumentParser(description='NNet loader')
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
	nnet = nn(g, nn_args)
	if args.input:
		nnet.load_checkpoint(os.path.dirname(args.input), os.path.basename(args.input))
	elif args.nn_version == -1:
		raise Exception("You have to specify at least a NN file to load or a NN version")

	from fvcore.nn import FlopCountAnalysis
	dummy_board         = torch.randn(1, g.getBoardSize()[0], g.getBoardSize()[1], dtype=torch.float32)
	dummy_valid_actions = torch.BoolTensor(torch.randn(1, g.getActionSize())>0.5)
	nnet.nnet.eval()
	flops = FlopCountAnalysis(nnet.nnet, (dummy_board, dummy_valid_actions))
	flops.unsupported_ops_warnings(False)
	# flops.uncalled_modules_warnings(False)
	print(f'V{nnet.nnet.version} -> {flops.total()/1000000:.1f} MFlops, nb params {nnet.number_params()[0]:.2e}')
	# print(flops.by_module().most_common(15))

	if not args.training:
		if args.input:
			checkpoint = torch.load(args.input, map_location='cpu')
			for k in sorted(checkpoint.keys()):
				if k not in ['state_dict', 'full_model', 'optim_state']:
					print(f'  {k}: {checkpoint[k]}')
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
