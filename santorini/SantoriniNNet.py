import torch
import torch.nn as nn
import torch.nn.functional as F

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

		if self.version == -1:
			pass # Special case when loading empty NN from pit.py

		elif self.version == 50:
			n_filters = 64

			self.first_layer = nn.Conv2d(  2, n_filters, 3, padding=1, bias=False)
			self.trunk = nn.Sequential(
				nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False), nn.BatchNorm2d(n_filters), nn.ReLU(),
				nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False), nn.BatchNorm2d(n_filters), nn.ReLU(),
				nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False), nn.BatchNorm2d(n_filters), nn.ReLU(),
				nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False), nn.BatchNorm2d(n_filters), nn.ReLU(),
			)

			self.output_layers_PI = nn.Sequential(
				nn.Linear(n_filters*5*5, 128),
				nn.Linear(128, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Linear(n_filters*5*5, 128),
				nn.Linear(128, self.num_players)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Linear(n_filters*5*5, 128),
				nn.Linear(128, self.num_scdiffs*self.scdiff_size)
			)

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
			sdiff = self.output_layers_SDIFF(x).squeeze(1)
			pi = torch.where(valid_actions, self.output_layers_PI(x).squeeze(1), self.lowvalue)

		elif self.version in [50]:
			x = input_data.transpose(-1, -2).view(-1, 3, 5, 5)
			x, data = x.split([2,1], dim=1)

			x = self.first_layer(x)
			x = self.trunk(x)
			x = torch.flatten(x, start_dim=1).unsqueeze(1)
			
			v = self.output_layers_V(x).squeeze(1)
			sdiff = self.output_layers_SDIFF(x).squeeze(1)
			pi = torch.where(valid_actions, self.output_layers_PI(x).squeeze(1), self.lowvalue)

		return F.log_softmax(pi, dim=1), torch.tanh(v), F.log_softmax(sdiff.view(-1, self.num_scdiffs, self.scdiff_size).transpose(1,2), dim=1) # TODO
