import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _make_divisible
from torchvision.models.mobilenetv3 import InvertedResidualConfig, InvertedResidual

from .AkropolisConstants import N_COLORS, CITY_SIZE

class LinearNormActivation(nn.Module):
	def __init__(self, in_size, out_size, activation_layer, depthwise=False, channels=None):
		super().__init__()
		self.linear     = nn.Linear(in_size, out_size, bias=False)
		self.norm       = nn.BatchNorm1d(channels if depthwise else out_size)
		self.activation = activation_layer(inplace=True) if activation_layer is not None else nn.Identity()
		self.depthwise = depthwise

	def forward(self, input):
		if self.depthwise:
			result = self.linear(input)
		else:
			result = self.linear(input.transpose(-1, -2)).transpose(-1, -2)
			
		result = self.norm(result)
		result = self.activation(result)
		return result

class SqueezeExcitation1d(nn.Module):
	def __init__(self, input_channels, squeeze_channels, scale_activation, setype='avg'):
		super().__init__()
		if setype == 'avg':
			self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
		else:
			self.avgpool = torch.nn.AdaptiveMaxPool1d(1)
		self.fc1 = nn.Linear(input_channels, squeeze_channels)
		self.activation = nn.ReLU()
		self.fc2 = torch.nn.Linear(squeeze_channels, input_channels)
		self.scale_activation = scale_activation()

	def _scale(self, input):
		scale = self.avgpool(input)
		scale = self.fc1(scale.transpose(-1, -2)).transpose(-1, -2)
		scale = self.activation(scale)
		scale = self.fc2(scale.transpose(-1, -2)).transpose(-1, -2)
		return self.scale_activation(scale)

	def forward(self, input):

		scale = self._scale(input)
		return scale * input

class InvertedResidual1d(nn.Module):
	def __init__(self, in_channels, exp_channels, out_channels, kernel, use_hs, use_se, setype='avg'):
		super().__init__()

		self.use_res_connect = (in_channels == out_channels)

		layers = []
		activation_layer = nn.Hardswish if use_hs else nn.ReLU

		# expand
		if exp_channels != in_channels:
			self.expand = LinearNormActivation(in_channels, exp_channels, activation_layer=activation_layer)
		else:
			self.expand = nn.Identity()

		# depthwise
		self.depthwise = LinearNormActivation(kernel, kernel, activation_layer=activation_layer, depthwise=True, channels=exp_channels)

		if use_se:
			squeeze_channels = _make_divisible(exp_channels // 4, 8)
			self.se = SqueezeExcitation1d(exp_channels, squeeze_channels, scale_activation=nn.Hardsigmoid, setype=setype)
		else:
			self.se = nn.Identity()

		# project
		self.project = LinearNormActivation(exp_channels, out_channels, activation_layer=None)

	def forward(self, input):
		result = self.expand(input)
		result = self.depthwise(result)
		result = self.se(result)
		result = self.project(result)

		if self.use_res_connect:
			result += input

		return result

class AkropolisNNet(nn.Module):
	def __init__(self, game, args):
		# game params
		# self.dim1, self.dim2, self.dim3 = game.getBoardSize()
		# self.dim1d_per_pl = N_COLORS
		# self.dim3d_per_pl = (CITY_SIZE, CITY_SIZE, 2)
	
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.args = args
		self.version = args['nn_version']
		super(AkropolisNNet, self).__init__()

		def inverted_residual(input_ch, expanded_ch, out_ch, use_se, activation, kernel=3, stride=1, dilation=1, width_mult=1):
			return InvertedResidual(InvertedResidualConfig(input_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation, width_mult), nn.BatchNorm2d)

		def filters_for_boards(input_ch, expanded_ch, out_ch, depth):
			filters  = [nn.Conv2d(input_ch, input_ch, kernel_size=3, padding=1, bias=False)]
			filters += [inverted_residual(input_ch, expanded_ch, input_ch, False, "RE") for i in range(depth//2)]
			filters += [inverted_residual(input_ch, expanded_ch, out_ch, False, "RE")]
			filters += [inverted_residual(out_ch, expanded_ch, out_ch, True, "HS") for i in range(depth//2)]
			filters += [nn.Flatten()]
			return nn.Sequential(*filters)

		if self.version == 1: # Ultra simple NN
			num_filters = 100
			self.trunk_1d = nn.Linear(2*CITY_SIZE*CITY_SIZE, num_filters)
			self.final_layers_V = nn.Linear(num_filters, 1)
			self.final_layers_PI = nn.Linear(num_filters, self.action_size)
		elif self.version == 10: # Very small version using MobileNetV3 building blocks
			### NN working on 3d data per player
			self.trunk_board = filters_for_boards(2, 8, N_COLORS)
			### NN working on 1d data + 0d data per player
			self.trunk_1d_per_pl = nn.Sequential(nn.Linear(CITY_SIZE, CITY_SIZE*2), nn.Linear(CITY_SIZE*2, CITY_SIZE))
			self.trunk_per_player = nn.Sequential(nn.Linear(N_COLORS+CITY_SIZE, CITY_SIZE*2), nn.Linear(CITY_SIZE*2, CITY_SIZE))
			### NN working on global data
			self.trunk_global = nn.Sequential(nn.Linear(2*CITY_SIZE*self.num_players+2*CITY_SIZE, 3*CITY_SIZE*self.num_players))
			self.final_layers_V = nn.Sequential(nn.Linear(3*CITY_SIZE*self.num_players, self.num_players), nn.Linear(self.num_players, 1))
			self.final_layers_PI = nn.Sequential(nn.Linear(3*CITY_SIZE*self.num_players, 3*self.action_size), nn.Linear(3*self.action_size, self.action_size))
	
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
		# Switch from NHWC to NCHW
		x = input_data.reshape(-1, CITY_SIZE, CITY_SIZE, 2*self.num_players+2).permute(0, 3, 1, 2)
		split_data = x.split([2]*self.num_players+[1,1], dim=1) # players board with 2 channels each, then per-player data, then global data
		inp_per_player_data = split_data[-2]
		global_data = split_data[-1]

		if self.version in [1]:
			# x.shape = Nx6x12x12
			# split_data[i].shape = Nx2x12x12
			x_1d = torch.cat([inp_per_player_data.flatten(1), global_data.flatten(1)], dim=1) # x_1d.shape = Nx288
			x_1d = self.trunk_1d(x_1d) # x_1d.shape = Nx100
			v = self.final_layers_V(x_1d)
			pi = torch.where(valid_actions, self.final_layers_PI(x_1d), self.lowvalue)
		elif self.version in [10]:
			list_per_player_1d = []
			for p in range(self.num_players):
				x_1d_board_pl = self.trunk_board(split_data[2*p:2*p+2])                 # Nx2xSIZExSIZE -> NxNCOLORS
				x_1d_data_pl    = self.trunk_1d_per_pl(inp_per_player_data[:, 0, p, :]) # NxSIZE -> NxSIZE
				x_1d_pl = self.trunk_per_player(torch.cat([x_1d_board_pl, x_1d_data_pl], dim=1))   # NxNCOLORS+SIZE -> Nx2SIZE
				list_per_player_1d.append(x_1d_pl)

			x_glob_1d = self.trunk_global(
				torch.cat(list_per_player_1d + [global_data[:, 0, 0, :], global_data[:, 0, 2, :]], dim=1),
			) # Nx(2*SIZE*NPLAYERS+SIZE+SIZE) -> NxZ

			v = self.final_layers_V(x_glob_1d)  # NxZ -> N
			pi = torch.where(valid_actions, self.final_layers_PI(x_glob_1d), self.lowvalue) # NxZ -> NxY

		else:
			raise Exception(f'Unsupported NN version {self.version}')

		return F.log_softmax(pi, dim=1), torch.tanh(v)

