import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _make_divisible
from torchvision.models.mobilenetv3 import InvertedResidualConfig, InvertedResidual

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

class BotanikNNet(nn.Module):
	def __init__(self, game, args):
		# game params
		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.nb_vect_1d = 6*5
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.args = args
		self.version = args['nn_version']
		super(BotanikNNet, self).__init__()

		def inverted_residual(input_ch, expanded_ch, out_ch, use_se, activation, kernel=3, stride=1, dilation=1, width_mult=1):
			return InvertedResidual(InvertedResidualConfig(input_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation, width_mult), nn.BatchNorm2d)

		if self.version == 10: # Very small version using MobileNetV3 building blocks
			### NN working on 1d data
			self.first_layer_1d = LinearNormActivation(self.vect_dim, self.vect_dim, None)
			confs = [InvertedResidual1d(self.vect_dim, 3*self.vect_dim, self.vect_dim, self.nb_vect_1d, False, "RE")]
			self.trunk_1d = nn.Sequential(*confs)

			head_PI_1d = [
				InvertedResidual1d(self.vect_dim, 3*self.vect_dim, self.vect_dim, self.nb_vect_1d, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(self.vect_dim*self.nb_vect_1d, self.action_size),
			]
			self.output_layers_PI_1d = nn.Sequential(*head_PI_1d)
			
			head_V_1d = [
				InvertedResidual1d(self.vect_dim, 3*self.vect_dim, self.vect_dim, self.nb_vect_1d, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(self.vect_dim*self.nb_vect_1d, self.num_players),
			]
			self.output_layers_V_1d = nn.Sequential(*head_V_1d)

			### NN working on current player's machine (2d)
			n_filters = 32
			n_exp_end = n_filters*3
			depth = 6 - 2

			self.first_layer_mach0 = nn.Conv2d(  7, n_filters, 3, padding=1, bias=False)
			confs  = [inverted_residual(n_filters, n_exp_end, n_filters, False, "RE") for i in range(depth//2)]
			self.trunk_mach0 = nn.Sequential(*confs)

			head_depth = 3
			n_exp_head = n_filters * 3
			head_PI_mach0 = [inverted_residual(n_filters, n_exp_head, n_filters, True, "HS",) for i in range(head_depth)] + [
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.action_size),
			]
			self.output_layers_PI_mach0 = nn.Sequential(*head_PI_mach0)
			head_V_mach0 = [inverted_residual(n_filters, n_exp_head, n_filters, True, "HS",) for i in range(head_depth)] + [
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.num_players),
			]
			self.output_layers_V_mach0 = nn.Sequential(*head_V_mach0)

			### NN working on other player's machine (2d)
			n_filters = 32
			n_exp_end = n_filters*3
			depth = 6 - 2

			self.first_layer_mach1 = nn.Conv2d(  7, n_filters, 3, padding=1, bias=False)
			confs  = [inverted_residual(n_filters, n_exp_end, n_filters, False, "RE") for i in range(depth//2)]
			self.trunk_mach1 = nn.Sequential(*confs)

			head_depth = 3
			n_exp_head = n_filters * 3
			head_PI_mach1 = [inverted_residual(n_filters, n_exp_head, n_filters, True, "HS",) for i in range(head_depth)] + [
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.action_size),
			]
			self.output_layers_PI_mach1 = nn.Sequential(*head_PI_mach1)
			head_V_mach1 = [inverted_residual(n_filters, n_exp_head, n_filters, True, "HS",) for i in range(head_depth)] + [
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.num_players),
			]
			self.output_layers_V_mach1 = nn.Sequential(*head_V_mach1)

			# Final output layers
			self.final_layers_PI = nn.Sequential(
				nn.Linear(self.action_size, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			)
			self.final_layers_V = nn.Sequential(
				nn.Linear(self.num_players, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			)

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
		x = input_data.transpose(-1, -2).view(-1, self.vect_dim, self.nb_vect)
		x = x[:,:,:16*5] # Remove access to some channels
		x_1d, x_mach0, x_mach1 = x.split([6*5,5*5,5*5], dim=2)

		x_mach0 = x_mach0.view(-1, self.vect_dim, 5, 5)
		x_mach1 = x_mach1.view(-1, self.vect_dim, 5, 5)

		if self.version in [10, 11]:
			x_1d = self.first_layer_1d(x_1d)
			x_1d = F.dropout(self.trunk_1d(x_1d), p=self.args['dropout'], training=self.training)
			pi_1d, v_1d = self.output_layers_PI_1d(x_1d), self.output_layers_V_1d(x_1d)

			x_mach0 = self.first_layer_mach0(x_mach0)
			x_mach0 = self.trunk_mach0(x_mach0)
			pi_mach0, v_mach0 = self.output_layers_PI_mach0(x_mach0), self.output_layers_V_mach0(x_mach0)

			x_mach1 = self.first_layer_mach1(x_mach1)
			x_mach1 = self.trunk_mach1(x_mach1)
			pi_mach1, v_mach1 = self.output_layers_PI_mach1(x_mach1), self.output_layers_V_mach1(x_mach1)

			v = self.final_layers_V(v_1d + v_mach0 + v_mach1)
			pi = torch.where(valid_actions, self.final_layers_PI(pi_1d + pi_mach0 + pi_mach1), self.lowvalue)

		else:
			raise Exception(f'Unsupported NN version {self.version}')

		return F.log_softmax(pi, dim=1), torch.tanh(v)

