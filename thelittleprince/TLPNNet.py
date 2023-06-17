import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _make_divisible

# Assume 3-dim tensor input N,C,L
class DenseAndPartialGPool(nn.Module):
	def __init__(self, input_length, output_length, nb_groups=8, nb_items_in_groups=8, channels_for_batchnorm=0):
		super().__init__()
		self.nb_groups = nb_groups
		self.nb_items_in_groups = nb_items_in_groups
		self.dense_input = input_length - nb_groups*nb_items_in_groups
		self.dense_output = output_length - 2*nb_groups
		self.dense_part = nn.Sequential(
			nn.Linear(self.dense_input, self.dense_output),
			nn.BatchNorm1d(channels_for_batchnorm) if channels_for_batchnorm > 0 else nn.Identity()
		)
		self.maxpool = nn.MaxPool1d(nb_items_in_groups)
		self.avgpool = nn.AvgPool1d(nb_items_in_groups)

	def forward(self, x):
		groups_for_gpool = x.split([self.nb_items_in_groups] * self.nb_groups + [self.dense_input], -1)
		maxpool_results = [ self.maxpool(y) for y in groups_for_gpool[:-1] ]
		avgpool_results = [ self.avgpool(y) for y in groups_for_gpool[:-1] ]
		
		dense_result = F.relu(self.dense_part(groups_for_gpool[-1]))

		x = torch.cat(maxpool_results + avgpool_results + [dense_result], -1)
		return x

# Assume 3-dim tensor input N,C,L, return N,1,L tensor
class FlattenAndPartialGPool(nn.Module):
	def __init__(self, length_to_pool, nb_channels_to_pool):
		super().__init__()
		self.length_to_pool = length_to_pool
		self.nb_channels_to_pool = nb_channels_to_pool
		self.maxpool = nn.MaxPool1d(nb_channels_to_pool)
		self.avgpool = nn.AvgPool1d(nb_channels_to_pool)
		self.flatten = nn.Flatten()

	def forward(self, x):
		x_begin, x_end = x[:,:,:self.length_to_pool], x[:,:,self.length_to_pool:]
		x_begin_firstC, x_begin_lastC = x_begin[:,:self.nb_channels_to_pool,:], x_begin[:,self.nb_channels_to_pool:,:]
		# MaxPool1D only applies to last dimension, whereas we want to apply on C dimension here
		x_begin_firstC = x_begin_firstC.transpose(-1, -2)
		maxpool_result = self.maxpool(x_begin_firstC).transpose(-1, -2)
		avgpool_result = self.avgpool(x_begin_firstC).transpose(-1, -2)
		x = torch.cat([
			self.flatten(maxpool_result),
			self.flatten(avgpool_result),
			self.flatten(x_begin_lastC),
			self.flatten(x_end)
		], 1)
		return x.unsqueeze(1)


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
		# print(f'Input -> {input.shape}')
		result = self.expand(input)
		# print(f'Expand -> {result.shape}')
		result = self.depthwise(result)
		# print(f'Depthwise -> {result.shape}')
		result = self.se(result)
		# print(f'SqEx -> {result.shape}')
		result = self.project(result)
		# print(f'Project -> {result.shape}')

		if self.use_res_connect:
			result += input

		# print(f'Result -> {result.shape}')
		return result




class TLPNNet(nn.Module):
	def __init__(self, game, args):
		# game params
		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.args = args
		self.version = args['nn_version']

		self.scdiff_size = 2 * (64-0) + 1
		self.num_scdiffs = self.num_players

		super(TLPNNet, self).__init__()
		if self.version == 1 or self.version == 398:
			self.dense2d_1 = nn.Sequential(
				nn.Linear(self.nb_vect, 128), nn.BatchNorm1d(15), nn.ReLU(),
				nn.Linear(128, 128)                            , nn.ReLU(), # no batchnorm before max pooling
			)

			self.partialgpool_1 = DenseAndPartialGPool(128, 128, nb_groups=4, nb_items_in_groups=8, channels_for_batchnorm=15)

			self.dense2d_2 = nn.Identity()
			self.partialgpool_2 = nn.Identity()

			self.dense2d_3 = nn.Sequential(
				nn.Linear(128, 128)                   , nn.ReLU(), # no batchnorm before max pooling
			)
			self.flatten_and_gpool = FlattenAndPartialGPool(length_to_pool=64, nb_channels_to_pool=15)
			self.dense1d_4 = nn.Sequential(
				nn.Linear(64*2+(128-64)*15, 128), nn.ReLU(),
			)
			self.partialgpool_4 = DenseAndPartialGPool(128, 128, nb_groups=4, nb_items_in_groups=4, channels_for_batchnorm=1)
			
			self.dense1d_5 = nn.Sequential(
				nn.Linear(128, 128), nn.BatchNorm1d(1), nn.ReLU(),
				nn.Linear(128, 128)                   , nn.ReLU(), # no batchnorm before max pooling
			)
			self.partialgpool_5 = DenseAndPartialGPool(128, 128, nb_groups=4, nb_items_in_groups=4, channels_for_batchnorm=1)

			self.output_layers_PI = nn.Sequential(
				nn.Linear(128, 128),
				nn.Linear(128, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Linear(128, 128),
				nn.Linear(128, self.num_players)
			)

			if self.version == 398:
				self.output_layers_SDIFF = nn.Sequential(
					nn.Linear(128, 128),
					nn.Linear(128, self.num_scdiffs*self.scdiff_size)
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
		if self.version == 1 or self.version == 398:
			x = input_data.transpose(-1, -2).view(-1, self.vect_dim, self.nb_vect)
			
			x = self.dense2d_1(x)
			x = F.dropout(self.partialgpool_1(x), p=self.args['dropout'], training=self.training)
			x = F.dropout(self.dense2d_3(x), p=self.args['dropout'], training=self.training)
			x = self.flatten_and_gpool(x)
			x = F.dropout(self.dense1d_4(x)     , p=self.args['dropout'], training=self.training)
			x = F.dropout(self.partialgpool_4(x), p=self.args['dropout'], training=self.training)
			x = F.dropout(self.dense1d_5(x)     , p=self.args['dropout'], training=self.training)
			x = F.dropout(self.partialgpool_5(x), p=self.args['dropout'], training=self.training)
			
			v = self.output_layers_V(x).squeeze(1)
			pi = torch.where(valid_actions, self.output_layers_PI(x).squeeze(1), self.lowvalue)

			if self.version == 398:
				sdiff = self.output_layers_SDIFF(x).squeeze(1)
				return F.log_softmax(pi, dim=1), torch.tanh(v), F.log_softmax(sdiff.view(-1, self.num_scdiffs, self.scdiff_size).transpose(1,2), dim=1) # TODO
		else:
			raise Exception(f'Unsupported NN version {self.version}')

		return F.log_softmax(pi, dim=1), torch.tanh(v)
