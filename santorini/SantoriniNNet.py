import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
from torchvision.models.mobilenetv3 import InvertedResidualConfig, InvertedResidual

class CatInTheMiddle(nn.Module):
	def __init__(self, layers2D, layers1D):
		super().__init__()
		self.layers2D, self.layers1D = layers2D, layers1D

	def forward(self, input2D, input1D):
		x = self.layers2D(input2D)
		x = torch.cat([torch.flatten(x, 1), torch.flatten(input1D, 1)], dim=-1)
		x = self.layers1D(x)
		return x


class SantoriniNNet(nn.Module):
	def __init__(self, game, args):
		# game params
		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.num_players = 2
		self.args = args
		self.version = args['nn_version']

		super(SantoriniNNet, self).__init__()

		if self.version == -1:
			pass # Special case when loading empty NN from pit.py
		elif self.version == 66:
			n_filters = 32
			n_filters_first = n_filters//2
			n_filters_begin = n_filters_first
			n_filters_end = n_filters
			n_exp_begin, n_exp_end = n_filters_begin*3, n_filters_end*3
			depth = 12 - 2

			def inverted_residual(input_ch, expanded_ch, out_ch, use_se, activation, kernel=3, stride=1, dilation=1, width_mult=1):
				return InvertedResidual(InvertedResidualConfig(input_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation, width_mult), nn.BatchNorm2d)
			self.first_layer = nn.Conv2d(  2, n_filters_first, 3, padding=1, bias=False)
			confs  = [inverted_residual(n_filters_first if i==0 else n_filters_begin, n_exp_begin, n_filters_begin, False, "RE") for i in range(depth//2)]
			confs += [inverted_residual(n_filters_begin if i==0 else n_filters_end  , n_exp_end  , n_filters_end  , True , "HS") for i in range(depth//2)]
			confs += [inverted_residual(n_filters_end                               , n_exp_end  , n_filters      , True , "HS")]
			self.trunk = nn.Sequential(*confs)

			head_depth = 6
			n_exp_head = n_filters * 3
			head_PI = [inverted_residual(n_filters, n_exp_head, n_filters, True, "HS",) for i in range(head_depth)] + [
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size)
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [inverted_residual(n_filters, n_exp_head, n_filters, True, "HS",) for i in range(head_depth)] + [
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players)
			]
			self.output_layers_V = nn.Sequential(*head_V)

		elif self.version == 67:
			n_filters = 32
			n_filters_first = n_filters//2
			n_filters_begin = n_filters_first
			n_filters_end = n_filters
			n_exp_begin, n_exp_end = n_filters_begin*3, n_filters_end*3
			depth = 12 - 2

			def inverted_residual(input_ch, expanded_ch, out_ch, use_se, activation, kernel=3, stride=1, dilation=1, width_mult=1):
				return InvertedResidual(InvertedResidualConfig(input_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation, width_mult), nn.BatchNorm2d)
			self.first_layer = nn.Conv2d(  2, n_filters_first, 3, padding=1, bias=False)
			confs  = [inverted_residual(n_filters_first if i==0 else n_filters_begin, n_exp_begin, n_filters_begin, False, "RE") for i in range(depth//2)]
			confs += [inverted_residual(n_filters_begin if i==0 else n_filters_end  , n_exp_end  , n_filters_end  , True , "HS") for i in range(depth//2)]
			confs += [inverted_residual(n_filters_end                               , n_exp_end  , n_filters      , True , "HS")]
			self.trunk = nn.Sequential(*confs)

			head_depth = 6
			n_exp_head = n_filters * 3
			head_PI_2D = [inverted_residual(n_filters, n_exp_head, n_filters, True, "HS",) for i in range(head_depth)]
			head_PI_1D = [
				nn.Linear((n_filters+1) *5*5, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size)
			]
			self.output_layers_PI = CatInTheMiddle(nn.Sequential(*head_PI_2D), nn.Sequential(*head_PI_1D))

			head_V_2D = [inverted_residual(n_filters, n_exp_head, n_filters, True, "HS",) for i in range(head_depth)]
			head_V_1D = [
				nn.Linear((n_filters+1) *5*5, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players)
			]
			self.output_layers_V = CatInTheMiddle(nn.Sequential(*head_V_2D), nn.Sequential(*head_V_1D))

		else:
			raise Exception(f'Warning, unknown NN version {self.version}')

		self.register_buffer('lowvalue', torch.FloatTensor([-1e8]))
		def _init(m):
			if type(m) == nn.Linear:
				nn.init.kaiming_uniform_(m.weight)
				nn.init.zeros_(m.bias)
			elif type(m) == nn.Sequential:
				for module in m:
					_init(module)
		for _, layer in self.__dict__.items():
			if isinstance(layer, nn.Module):
				layer.apply(_init)

	def forward(self, input_data, valid_actions):
		if self.version == 1:
			x = input_data.transpose(-1, -2).view(-1, self.vect_dim, self.nb_vect)
		
			x = torch.flatten(x, start_dim=1).unsqueeze(1)
			x = F.dropout(self.dense1d_1(x)     , p=self.args['dropout'], training=self.training)
			x = F.dropout(self.partialgpool_1(x), p=self.args['dropout'], training=self.training)
			x = F.dropout(self.dense1d_2(x)     , p=self.args['dropout'], training=self.training)
			x = F.dropout(self.partialgpool_2(x), p=self.args['dropout'], training=self.training)
			x = F.dropout(self.dense1d_3(x)     , p=self.args['dropout'], training=self.training)
			
			v = self.output_layers_V(x).squeeze(1)
			pi = torch.where(valid_actions, self.output_layers_PI(x).squeeze(1), self.lowvalue)

		elif self.version in [66]:
			x = input_data.transpose(-1, -2).view(-1, 3, 5, 5)
			x, data = x.split([2,1], dim=1)
			x = self.first_layer(x)
			x = self.trunk(x)
			v = self.output_layers_V(x)
			pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)

		elif self.version in [67]:
			x = input_data.transpose(-1, -2).view(-1, 3, 5, 5)
			x, data = x.split([2,1], dim=1)
			x = self.first_layer(x)
			x = self.trunk(x)
			v = self.output_layers_V(x, data)
			pi = torch.where(valid_actions, self.output_layers_PI(x, data), self.lowvalue)

		return F.log_softmax(pi, dim=1), torch.tanh(v)
