import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _make_divisible

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



class SmallworldNNet(nn.Module):
	def __init__(self, game, args):
		# game params
		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.args = args
		self.version = args['nn_version']

		super(SmallworldNNet, self).__init__()
		if self.version == 10: # Small but wide
			self.first_layer = LinearNormActivation(self.nb_vect, 64, None)
			confs  = []
			confs += [InvertedResidual1d(64, 192, 64, 8, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			n_filters = 64
			head_PI = [
				InvertedResidual1d(64, 192, 64, 8, True, "HS", setype='avg'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(64, 192, 64, 8, True, "HS", setype='avg'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

		elif self.version == 11: # Same as 10 but with max for S&E instead of avg
			self.first_layer = LinearNormActivation(self.nb_vect, 64, None)
			confs  = []
			confs += [InvertedResidual1d(64, 192, 64, 8, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			n_filters = 64
			head_PI = [
				InvertedResidual1d(64, 192, 64, 8, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(64, 192, 64, 8, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

		elif self.version == 20: # Like V10 but bigger in all dimensions
			self.first_layer = LinearNormActivation(self.nb_vect, 128, None)
			confs  = []
			confs += [InvertedResidual1d(128, 192, 128, 8, False, "RE")]
			confs += [InvertedResidual1d(128, 192, 128, 8, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			n_filters = 64
			head_PI = [
				InvertedResidual1d(128, 192, 128, 8, True, "HS", setype='avg'),
				InvertedResidual1d(128, 192,  64, 8, True, "HS", setype='avg'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(128, 192, 128, 8, True, "HS", setype='avg'),
				InvertedResidual1d(128, 192,  64, 8, True, "HS", setype='avg'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

		elif self.version == 21: # Like V20 but with 'max' instead of 'avg'
			self.first_layer = LinearNormActivation(self.nb_vect, 128, None)
			confs  = []
			confs += [InvertedResidual1d(128, 192, 128, 8, False, "RE")]
			confs += [InvertedResidual1d(128, 192, 128, 8, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			n_filters = 64
			head_PI = [
				InvertedResidual1d(128, 192, 128, 8, True, "HS", setype='max'),
				InvertedResidual1d(128, 192,  64, 8, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(128, 192, 128, 8, True, "HS", setype='max'),
				InvertedResidual1d(128, 192,  64, 8, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

		elif self.version == 31: # Like V21 but in bigger
			self.first_layer = LinearNormActivation(self.nb_vect, 256, None)
			confs  = []
			confs += [InvertedResidual1d(256, 384, 256, 8, False, "RE")]
			confs += [InvertedResidual1d(256, 384, 256, 8, False, "RE")]
			confs += [InvertedResidual1d(256, 384, 256, 8, False, "RE")]
			confs += [InvertedResidual1d(256, 384, 256, 8, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			n_filters = 128
			head_PI = [
				InvertedResidual1d(256, 384, 256, 8, True, "HS", setype='avg'),
				InvertedResidual1d(256, 384, 256, 8, True, "HS", setype='max'),
				InvertedResidual1d(256, 384, 256, 8, True, "HS", setype='max'),
				InvertedResidual1d(256, 384, 128, 8, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, n_filters *8),
				nn.ReLU(),
				nn.Linear(n_filters *8, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(256, 384, 256, 8, True, "HS", setype='avg'),
				InvertedResidual1d(256, 384, 256, 8, True, "HS", setype='max'),
				InvertedResidual1d(256, 384, 256, 8, True, "HS", setype='max'),
				InvertedResidual1d(256, 384, 128, 8, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, n_filters *8),
				nn.ReLU(),
				nn.Linear(n_filters *8, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

		elif self.version == 32: # Like V31 but narrower
			self.first_layer = LinearNormActivation(self.nb_vect, 128, None)
			confs  = []
			confs += [InvertedResidual1d(128, 192, 128, 8, False, "RE")]
			confs += [InvertedResidual1d(128, 192, 128, 8, False, "RE")]
			confs += [InvertedResidual1d(128, 192, 128, 8, False, "RE")]
			confs += [InvertedResidual1d(128, 192, 128, 8, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			n_filters = 64
			head_PI = [
				InvertedResidual1d(128, 192, 128, 8, True, "HS", setype='avg'),
				InvertedResidual1d(128, 192, 128, 8, True, "HS", setype='max'),
				InvertedResidual1d(128, 192, 128, 8, True, "HS", setype='max'),
				InvertedResidual1d(128, 192, 64, 8, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, n_filters *8),
				nn.ReLU(),
				nn.Linear(n_filters *8, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(128, 192, 128, 8, True, "HS", setype='avg'),
				InvertedResidual1d(128, 192, 128, 8, True, "HS", setype='max'),
				InvertedResidual1d(128, 192, 128, 8, True, "HS", setype='max'),
				InvertedResidual1d(128, 192, 64, 8, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, n_filters *8),
				nn.ReLU(),
				nn.Linear(n_filters *8, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

		elif self.version == 33: # Like V31 but narrower
			self.first_layer = LinearNormActivation(self.nb_vect, 128, None)
			confs  = []
			confs += [InvertedResidual1d(128, 384, 128, 8, False, "RE")]
			confs += [InvertedResidual1d(128, 384, 128, 8, False, "RE")]
			confs += [InvertedResidual1d(128, 384, 128, 8, False, "RE")]
			confs += [InvertedResidual1d(128, 384, 128, 8, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			n_filters = 64
			head_PI = [
				InvertedResidual1d(128, 384, 128, 8, True, "HS", setype='avg'),
				InvertedResidual1d(128, 384, 128, 8, True, "HS", setype='max'),
				InvertedResidual1d(128, 384, 128, 8, True, "HS", setype='max'),
				InvertedResidual1d(128, 384, 64, 8, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, n_filters *8),
				nn.ReLU(),
				nn.Linear(n_filters *8, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(128, 384, 128, 8, True, "HS", setype='avg'),
				InvertedResidual1d(128, 384, 128, 8, True, "HS", setype='max'),
				InvertedResidual1d(128, 384, 128, 8, True, "HS", setype='max'),
				InvertedResidual1d(128, 384, 64, 8, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, n_filters *8),
				nn.ReLU(),
				nn.Linear(n_filters *8, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

		elif self.version == 34: # Like V31 but shallower
			self.first_layer = LinearNormActivation(self.nb_vect, 256, None)
			confs  = []
			confs += [InvertedResidual1d(256, 384, 256, 8, False, "RE")]
			confs += [InvertedResidual1d(256, 384, 256, 8, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			n_filters = 128
			head_PI = [
				InvertedResidual1d(256, 384, 256, 8, True, "HS", setype='max'),
				InvertedResidual1d(256, 384, 128, 8, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, n_filters *8),
				nn.ReLU(),
				nn.Linear(n_filters *8, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(256, 384, 256, 8, True, "HS", setype='max'),
				InvertedResidual1d(256, 384, 128, 8, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, n_filters *8),
				nn.ReLU(),
				nn.Linear(n_filters *8, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

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
		if self.version in [10, 11, 20, 21, 31, 32, 33, 34]: # Use input as is
			x = input_data.view(-1, self.nb_vect, self.vect_dim) # no transpose
			x = self.first_layer(x)
			x = F.dropout(self.trunk(x), p=self.args['dropout'], training=self.training)
			v = self.output_layers_V(x)
			pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)

		else:
			raise Exception(f'Unsupported NN version {self.version}')

		return F.log_softmax(pi, dim=1), torch.tanh(v)

