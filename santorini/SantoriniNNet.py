import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SantoriniNNet(nn.Module):
	def __init__(self, game, args):
		# game params
		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.scdiff_size = 2 * game.getMaxScoreDiff() + 1
		self.num_players = 2
		self.num_scdiffs = 2 # Number of combinations of 2 players
		self.args = args
		self.version = args['nn_version']

		super(SantoriniNNet, self).__init__()

		self.dense1d_1 = nn.Sequential(
			nn.Linear(self.nb_vect*self.vect_dim, 128), nn.ReLU(),
		)
		self.partialgpool_1 = DenseAndPartialGPool(128, 128, nb_groups=6, nb_items_in_groups=3, channels_for_batchnorm=1)
		
		self.dense1d_2 = nn.Sequential(
			nn.Linear(128, 128), nn.BatchNorm1d(1), nn.ReLU(),
			nn.Linear(128, 128)                   , nn.ReLU(), # no batchnorm before max pooling
		)
		self.partialgpool_2 = DenseAndPartialGPool(128, 128, nb_groups=6, nb_items_in_groups=3, channels_for_batchnorm=1)

		self.dense1d_3 = nn.Sequential(
			nn.Linear(128, 128), nn.BatchNorm1d(1), nn.ReLU(),
			nn.Linear(128, 128), nn.BatchNorm1d(1), nn.ReLU(),
		)

		self.output_layers_PI = nn.Sequential(
			nn.Linear(128, 128),
			nn.Linear(128, self.action_size)
		)

		self.output_layers_V = nn.Sequential(
			nn.Linear(128, 128),
			nn.Linear(128, self.num_players)
		)

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
		for layer1D in [self.dense1d_1, self.partialgpool_1, self.dense1d_2, self.dense1d_3, self.partialgpool_2, self.output_layers_PI, self.output_layers_V, self.output_layers_SDIFF]:
			layer1D.apply(_init)

	def forward(self, input_data, valid_actions):
		x = input_data.transpose(-1, -2).view(-1, self.vect_dim, self.nb_vect)
		
		x = torch.flatten(x, start_dim=1).unsqueeze(1)
		x = F.dropout(self.dense1d_1(x)     , p=self.args['dropout'], training=self.training)
		x = F.dropout(self.partialgpool_1(x), p=self.args['dropout'], training=self.training)
		x = F.dropout(self.dense1d_2(x)     , p=self.args['dropout'], training=self.training)
		x = F.dropout(self.partialgpool_2(x), p=self.args['dropout'], training=self.training)
		x = F.dropout(self.dense1d_3(x)     , p=self.args['dropout'], training=self.training)
		
		v = self.output_layers_V(x).squeeze(1)
		sdiff = self.output_layers_SDIFF(x).squeeze(1)
		pi = torch.where(valid_actions, self.output_layers_PI(x).squeeze(1), self.lowvalue)

		return F.log_softmax(pi, dim=1), torch.tanh(v), F.log_softmax(sdiff.view(-1, self.num_scdiffs, self.scdiff_size).transpose(1,2), dim=1) # TODO
