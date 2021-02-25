import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import torch
import torch.optim as optim

from .SplendorNNet import SplendorNNet as snnet

def get_uptime():
	import subprocess
	tuptime = subprocess.run(['tuptime', '--power', '--seconds', '--csv'], capture_output=True)
	tuptime_stdout = tuptime.stdout.decode('utf-8')
	runtime_value = int(tuptime_stdout.splitlines()[3].split(',')[-1].split(' ')[1])
	return runtime_value

class NNetWrapper(NeuralNet):
	def __init__(self, game, nn_args):
		self.args = nn_args
		self.args['cuda'] = torch.cuda.is_available()
		self.nnet = snnet(game, nn_args)
		
		# PRINT MODEL #
		# print(self.nnet)
		# for name, parameter in self.nnet.named_parameters():
		# 	if not parameter.requires_grad: continue
		# 	param = parameter.numel()
		# 	if param > 50000:
		# 		print(name, param)
		# print(sum([p.numel() for _, p in self.nnet.named_parameters()]))
		# EXPORT TO ONNX AND VIEW ON NETRON.APP #
		# example_input = torch.randn((64,53,7))
		# valids = torch.BoolTensor(np.array([True]*81).astype(np.bool_))
		# torch.onnx.export(self.nnet, (example_input, valids), 'splendornnet.onnx')
		# exit(42)

		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()

		self.begin_uptime = get_uptime()
		self.cumulated_uptime = 0
		self.begin_time = int(time.time())

		if self.args['cuda']:
			self.nnet.cuda()
		else:
			torch.set_num_threads(1) # CPU much more efficient when using 1 thread than severals

	def train(self, examples):
		"""
		examples: list of examples, each example is of form (board, pi, v)
		"""
		optimizer = optim.Adam(self.nnet.parameters())

		batch_count = int(len(examples) / self.args['batch_size'])

		t = tqdm(total=self.args['epochs'] * batch_count, desc='Train ep0', colour='blue', ncols=100, mininterval=0.5)
		for epoch in range(self.args['epochs']):
			t.set_description(f'Train ep{epoch + 1}')
			self.nnet.train()
			pi_losses = AverageMeter()
			v_losses = AverageMeter()
	
			for _ in range(batch_count):
				sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
				boards, pis, vs, valid_actions = list(zip(*[examples[i] for i in sample_ids]))
				boards = torch.FloatTensor(np.array(boards).astype(np.float32))
				valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_))
				target_pis = torch.FloatTensor(np.array(pis).astype(np.float32))
				target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))

				# predict
				if self.args['cuda']:
					boards, target_pis, target_vs, valid_actions = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda(), valid_actions.contiguous().cuda()

				# compute output
				out_pi, out_v = self.nnet(boards, valid_actions)
				l_pi = self.loss_pi(target_pis, out_pi)
				l_v = self.loss_v(target_vs, out_v)
				total_loss = l_pi + l_v

				# record loss
				pi_losses.update(l_pi.item(), boards.size(0))
				v_losses.update(l_v.item(), boards.size(0))
				t.set_postfix(lossPI=pi_losses, lossV=v_losses, refresh=False)

				# compute gradient and do SGD step
				optimizer.zero_grad()
				total_loss.backward()
				optimizer.step()

				t.update()
		t.close()

	def predict(self, board, valid_actions):
		"""
		board: np array with board
		"""
		# timing

		# preparing input
		board = torch.FloatTensor(board.astype(np.float32))
		valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_))
		if self.args['cuda']:
			board         = board.contiguous().cuda()
			valid_actions = valid_actions.contiguous().cuda()
		self.nnet.eval()
		with torch.no_grad():
				pi, v = self.nnet(board, valid_actions)

		return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

	def loss_pi(self, targets, outputs):
		return -torch.sum(targets * outputs) / targets.size()[0]

	def loss_v(self, targets, outputs):
		return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

	def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
		filepath = os.path.join(folder, filename)
		if not os.path.exists(folder):
			# print("Checkpoint Directory does not exist! Making directory {}".format(folder))
			os.mkdir(folder)
		# else:
		#     print("Checkpoint Directory exists! ")
		current_uptime = get_uptime()
		torch.save({
			'state_dict': self.nnet.state_dict(),
			'full_model': self.nnet,
			'cumulated_uptime': self.cumulated_uptime + current_uptime-self.begin_uptime,
			'end_uptime': current_uptime,
			'begin': self.begin_time,
		}, filepath)

	def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', ongoing_experiment=False):
		# https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
		filepath = os.path.join(folder, filename)
		if not os.path.exists(filepath):
			print("No model in path {}".format(filepath))
			return
		try:
			map_location = None if self.args['cuda'] else 'cpu'
			checkpoint = torch.load(filepath, map_location=map_location)
		except:
			print("No model in path {} but file exists".format(filepath))
			return
		self.nnet = checkpoint['full_model']
		self.cumulated_uptime = checkpoint.get('cumulated_uptime', 0)
		self.begin_time = checkpoint.get('begin', int(time.time()))
		self.begin_uptime = checkpoint.get('end_uptime', 0) if ongoing_experiment else get_uptime()
			
