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
		self.device = {
			'training' : 'cpu', #'cuda' if torch.cuda.is_available() else 'cpu',
			'inference': 'onnx',
		}
		self.current_mode = 'cpu'
		self.nnet = snnet(game, nn_args)
		self.ort_session = None

		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.max_diff = game.getMaxScoreDiff()

		self.begin_uptime = get_uptime()
		self.cumulated_uptime = 0
		self.begin_time = int(time.time())

	def train(self, examples):
		"""
		examples: list of examples, each example is of form (board, pi, v)
		"""
		self.switch_target('training')

		optimizer = optim.Adam(self.nnet.parameters())
		batch_count = int(len(examples) / self.args['batch_size'])
		examples_surprises = np.array([x[5] for x in examples])
		examples_weights = examples_surprises / examples_surprises.sum() + 1./len(examples_surprises)
		examples_weights = examples_weights / examples_weights.sum()

		t = tqdm(total=self.args['epochs'] * batch_count, desc='Train ep0', colour='blue', ncols=100, mininterval=0.5, disable=None)
		for epoch in range(self.args['epochs']):
			t.set_description(f'Train ep{epoch + 1}')
			self.nnet.train()
			pi_losses, v_losses, scdiff_losses = AverageMeter(), AverageMeter(), AverageMeter()
	
			for _ in range(batch_count):
				# sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
				samples_ids = np.random.choice(len(examples), size=self.args['batch_size'], replace=False, p=examples_weights)
				boards, pis, vs, scdiffs, valid_actions, surprises = list(zip(*[examples[i] for i in sample_ids]))
				boards = torch.FloatTensor(np.array(boards).astype(np.float32))
				valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_))
				target_pis = torch.FloatTensor(np.array(pis).astype(np.float32))
				target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))
				target_scdiffs = torch.FloatTensor(np.zeros((len(scdiffs), 2*self.max_diff+1)).astype(np.float32))
				for i in range(len(scdiffs)):
					score_diff = min(max(scdiffs[i] + self.max_diff, 0), 2*self.max_diff)
					target_scdiffs[i, score_diff] = 1

				# predict
				if self.device['training'] == 'cuda':
					boards, target_pis, target_vs, valid_actions, target_scdiffs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda(), valid_actions.contiguous().cuda(), target_scdiffs.contiguous().cuda()

				# compute output
				out_pi, out_v, out_scdiff = self.nnet(boards, valid_actions)
				l_pi = self.loss_pi(target_pis, out_pi)
				l_v = self.loss_v(target_vs, out_v)
				l_scdiff_c = self.loss_scdiff_cdf(target_scdiffs, out_scdiff)
				l_scdiff_p = self.loss_scdiff_pdf(target_scdiffs, out_scdiff)
				total_loss = l_pi + l_v + l_scdiff_c + l_scdiff_p

				# record loss
				pi_losses.update(l_pi.item(), boards.size(0))
				v_losses.update(l_v.item(), boards.size(0))
				scdiff_losses.update(l_scdiff_c.item() + l_scdiff_p.item(), boards.size(0))
				t.set_postfix(PI=pi_losses, V=v_losses, SD=scdiff_losses, refresh=False)

				# compute gradient and do SGD step
				optimizer.zero_grad(set_to_none=True)
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
		self.switch_target('inference')

		if self.current_mode == 'onnx':
			ort_outs = self.ort_session.run(None, {
				'board': board.astype(np.float32).reshape((-1, self.nb_vect, self.vect_dim)),
				'valid_actions': np.array(valid_actions).astype(np.bool_).reshape((-1, self.action_size)),
			})
			return np.exp(ort_outs[0])[0], ort_outs[1][0]

		else:
			board = torch.FloatTensor(board.astype(np.float32))
			valid_actions = torch.BoolTensor(np.array(valid_actions).astype(np.bool_))
			if self.current_mode == 'cuda':
				board, valid_actions = board.contiguous().cuda(), valid_actions.contiguous().cuda()
			self.nnet.eval()
			with torch.no_grad():
				pi, v, _ = self.nnet(board, valid_actions)

			return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

	def loss_pi(self, targets, outputs):
		return -torch.sum(targets * outputs) / targets.size()[0]

	def loss_v(self, targets, outputs):
		return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

	def loss_scdiff_cdf(self, targets, outputs):
		l2_diff = torch.square(torch.cumsum(targets, axis=1) - torch.cumsum(torch.exp(outputs), axis=1))
		return 0.02 * torch.sum(l2_diff) / targets.size()[0]

	def loss_scdiff_pdf(self, targets, outputs):
		cross_entropy = -torch.sum( torch.mul(targets, outputs) )
		return 0.02 * cross_entropy / targets.size()[0]

	def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
		filepath = os.path.join(folder, filename)
		if not os.path.exists(folder):
			# print("Checkpoint Directory does not exist! Making directory {}".format(folder))
			os.mkdir(folder)
		# else:
		#     print("Checkpoint Directory exists! ")
		current_uptime = get_uptime()

		self.switch_target('inference')
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
			checkpoint = torch.load(filepath, map_location='cpu')
		except:
			print("No model in path {} but file exists".format(filepath))
			return
		self.nnet = checkpoint['full_model']
		self.cumulated_uptime = checkpoint.get('cumulated_uptime', 0)
		self.begin_time = checkpoint.get('begin', int(time.time()))
		self.begin_uptime = checkpoint.get('end_uptime', 0) if ongoing_experiment else get_uptime()
			
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
			opset_version=11, # the ONNX version to export the model to
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

		self.ort_session = ort.InferenceSession(temporary_file)
		os.remove(temporary_file)
