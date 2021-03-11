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


class SplendorNNet(nn.Module):
	def __init__(self, game, args):
		# game params
		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.scdiff_size = 2 * game.getMaxScoreDiff() + 1
		self.args = args
		self.version = args['nn_version']

		super(SplendorNNet, self).__init__()
		def _init(m):
			if type(m) == nn.Linear:
				nn.init.kaiming_uniform_(m.weight)
				nn.init.zeros_(m.bias)

		if self.version == 1:
			dense2d, dense1d = [256,256], [512,256,256]
			self.linear2D = nn.ModuleList([nn.Linear(prev, cur) for prev, cur in zip([self.nb_vect] +dense2d, dense2d)])
			self.linear2D.apply(_init)
			self.linear1D = nn.ModuleList([nn.Linear(prev, cur) for prev, cur in zip([dense2d[-1]*9]+dense1d, dense1d)])
			self.linear1D.apply(_init)
			self.batch_norms = nn.ModuleList([nn.BatchNorm1d(cur) for cur in dense1d])
	 
			self.output_layer_PI     = nn.Linear(dense1d[-1], self.action_size)
			self.output_layer_V      = nn.Linear(dense1d[-1], 1)
			self.output_layer_SDIFF  = nn.Linear(dense1d[-1], self.scdiff_size)
			self.maxpool = nn.MaxPool2d((5,1))
			self.avgpool = nn.AvgPool2d((5,1))

		elif self.version in [3, 8]:
			self.dense2d_1 = nn.Sequential(
				nn.Linear(self.nb_vect, 256), nn.BatchNorm1d(7), nn.ReLU(),
				nn.Linear(256, 256)                            , nn.ReLU(), # no batchnorm before max pooling
			)
			if self.version == 3:
				self.partialgpool_1 = DenseAndPartialGPool(256, 256, nb_groups=8, nb_items_in_groups=8, channels_for_batchnorm=7)
			else:
				self.partialgpool_1 = DenseAndPartialGPool(256, 256, nb_groups=4, nb_items_in_groups=8, channels_for_batchnorm=7)

			self.dense2d_2 = nn.Identity()
			self.partialgpool_2 = nn.Identity()

			self.dense2d_3 = nn.Sequential(
				nn.Linear(256, 128)                   , nn.ReLU(), # no batchnorm before max pooling
			)
			self.flatten_and_gpool = FlattenAndPartialGPool(length_to_pool=64, nb_channels_to_pool=5)

			self.dense1d_4 = nn.Sequential(
				nn.Linear(64*4+(128-64)*7, 256), nn.ReLU(),
			)
			if self.version == 3:
				self.partialgpool_4 = DenseAndPartialGPool(256, 256, nb_groups=8, nb_items_in_groups=4, channels_for_batchnorm=1)
			else:
				self.partialgpool_4 = DenseAndPartialGPool(256, 256, nb_groups=4, nb_items_in_groups=4, channels_for_batchnorm=1)

			self.dense1d_5 = nn.Sequential(
				nn.Linear(256, 128), nn.BatchNorm1d(1), nn.ReLU(),
				nn.Linear(128, 128)                   , nn.ReLU(), # no batchnorm before max pooling
			)
			self.partialgpool_5 = DenseAndPartialGPool(128, 128, nb_groups=4, nb_items_in_groups=4, channels_for_batchnorm=1)

			self.output_layers_PI = nn.Sequential(
				nn.Linear(128, 128),
				nn.Linear(128, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Linear(128, 128),
				nn.Linear(128, 1)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Linear(128, 128),
				nn.Linear(128, self.scdiff_size)
			)

		elif self.version == 9:
			self.dense2d_1 = nn.Sequential(
				nn.Linear(self.nb_vect, 512), nn.BatchNorm1d(7), nn.ReLU(),
				nn.Linear(512, 512)                            , nn.ReLU(), # no batchnorm before max pooling
			)
			self.partialgpool_1 = DenseAndPartialGPool(512, 512, nb_groups=8, nb_items_in_groups=8, channels_for_batchnorm=7)

			self.dense2d_2 = nn.Sequential(
				nn.Linear(512, 512), nn.BatchNorm1d(7), nn.ReLU(),
				nn.Linear(512, 512)                   , nn.ReLU(), # no batchnorm before max pooling
			)
			self.partialgpool_2 = DenseAndPartialGPool(512, 512, nb_groups=8, nb_items_in_groups=8, channels_for_batchnorm=7)

			self.dense2d_3 = nn.Sequential(
				nn.Linear(512, 256)                   , nn.ReLU(), # no batchnorm before max pooling
			)
			self.flatten_and_gpool = FlattenAndPartialGPool(length_to_pool=64, nb_channels_to_pool=5)

			self.dense1d_4 = nn.Sequential(
				nn.Linear(64*4+(256-64)*7, 512), nn.ReLU(),
			)
			self.partialgpool_4 = DenseAndPartialGPool(512, 512, nb_groups=8, nb_items_in_groups=4, channels_for_batchnorm=1)

			self.dense1d_5 = nn.Sequential(
				nn.Linear(512, 256), nn.BatchNorm1d(1), nn.ReLU(),
				nn.Linear(256, 256)                   , nn.ReLU(), # no batchnorm before max pooling
			)
			self.partialgpool_5 = DenseAndPartialGPool(256, 256, nb_groups=4, nb_items_in_groups=4, channels_for_batchnorm=1)

			self.output_layers_PI = nn.Sequential(
				nn.Linear(256, 256),
				nn.Linear(256, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Linear(256, 256),
				nn.Linear(256, 1)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Linear(256, 256),
				nn.Linear(256, self.scdiff_size)
			)


		if self.version >= 2:
			self.register_buffer('lowvalue', torch.FloatTensor([-1e8]))
			for layer2D in [self.dense2d_1, self.partialgpool_1, self.dense2d_3, self.flatten_and_gpool]:
				layer2D.apply(_init)
			for layer1D in [self.dense1d_4, self.partialgpool_4, self.dense1d_5, self.partialgpool_5, self.output_layers_PI, self.output_layers_V, self.output_layers_SDIFF]:
				layer1D.apply(_init)

	def forward(self, input_data, valid_actions):
		if self.version == 1:
			if len(input_data.shape) == 3:
				x = input_data.permute(0,2,1).view(-1, self.vect_dim, self.nb_vect)
			else:
				x = input_data.permute(1,0).view(-1, self.vect_dim, self.nb_vect)
			
			for layer2d in self.linear2D:
				x = F.relu(layer2d(x))
			
			x_5lay,_ = x.split([5,2], 1)
			x_max    = self.maxpool(x_5lay)
			x_avg    = self.avgpool(x_5lay)
			x = nn.Flatten()(torch.cat((x_max, x_avg, x), 1))

			for layer1d, bn in zip(self.linear1D, self.batch_norms):
				x = F.dropout(F.relu(bn(layer1d(x))), p=self.args['dropout'], training=self.training)
			
			v = self.output_layer_V(x)
			sdiff = self.output_layer_SDIFF(x)
			pi = torch.where(valid_actions, self.output_layer_PI(x), torch.FloatTensor([-1e8]))
			
			return F.log_softmax(pi, dim=1), torch.tanh(v), F.log_softmax(sdiff, dim=1),

		if self.version >= 2:
			x = input_data.transpose(-1, -2).view(-1, self.vect_dim, self.nb_vect)
			
			x = self.dense2d_1(x)
			x = self.partialgpool_1(x)
			if self.version == 9:
				x = self.dense2d_2(x)
				x = self.partialgpool_2(x)
			x = self.dense2d_3(x)
			x = self.flatten_and_gpool(x)
			x = F.dropout(self.dense1d_4(x)     , p=self.args['dropout'], training=self.training) 
			x = F.dropout(self.partialgpool_4(x), p=self.args['dropout'], training=self.training)
			x = F.dropout(self.dense1d_5(x)     , p=self.args['dropout'], training=self.training)
			x = F.dropout(self.partialgpool_5(x), p=self.args['dropout'], training=self.training)
			
			v = self.output_layers_V(x).squeeze(1)
			sdiff = self.output_layers_SDIFF(x).squeeze(1)
			pi = torch.where(valid_actions, self.output_layers_PI(x).squeeze(1), self.lowvalue)

			return F.log_softmax(pi, dim=1), torch.tanh(v), F.log_softmax(sdiff, dim=1)
