import torch
import torch.nn as nn
import torch.nn.functional as F

class Residualv1(nn.Module):
	def __init__(self, n_filters):
		super().__init__()
		self.residual = nn.Sequential(
			nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False),
			nn.BatchNorm2d(n_filters),
			nn.ReLU(),
			nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False),
			nn.BatchNorm2d(n_filters),
		)
		self.final = nn.ReLU()

	def forward(self, x):
		y = self.residual(x)
		x = y + x
		x = self.final(x)
		return x

class Residualv2(nn.Module):
	def __init__(self, n_filters):
		super().__init__()
		self.residual = nn.Sequential(
			nn.BatchNorm2d(n_filters),
			nn.ReLU(),
			nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False),
			nn.BatchNorm2d(n_filters),
			nn.ReLU(),
			nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False),
		)

	def forward(self, x):
		y = self.residual(x)
		x = y + x
		return x

class SE_Residualv2(nn.Module):
	def __init__(self, n_filters, reduction=16, pool='max'):
		super().__init__()
		self.residual = nn.Sequential(
			nn.BatchNorm2d(n_filters),
			nn.ReLU(),
			nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False),
			nn.BatchNorm2d(n_filters),
			nn.ReLU(),
			nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False),
		)
		self.pool = nn.MaxPool2d(5) if pool == 'max' else nn.AvgPool2d(5)
		self.fc = nn.Sequential(
			nn.Linear(n_filters, n_filters // reduction, bias=False),
			nn.ReLU(),
			nn.Linear(n_filters // reduction, n_filters, bias=False),
			nn.Sigmoid(),
		)

	def forward(self, x):
		y = self.residual(x)

		b, c, _, _ = y.size()
		z = self.pool(y).view(b, c)
		z = self.fc(z).view(b, c, 1, 1)
		y = y * z.expand_as(y)
		x = x + y

		return x


class InceptionA(nn.Module):
	def __init__(self, n_filters, pool='max'):
		super().__init__()
		self.branch1x1 = nn.Sequential(
			nn.Conv2d(n_filters, n_filters//4, 1, padding=0, bias=False),
			nn.BatchNorm2d(n_filters//4),
			nn.ReLU(),
		)

		self.branch5x5 = nn.Sequential(
			nn.Conv2d(n_filters, n_filters//4, 1, padding=0, bias=False),
			nn.BatchNorm2d(n_filters//4),
			nn.ReLU(),
			nn.Conv2d(n_filters//4, n_filters//4, 5, padding=2, bias=False),
			nn.BatchNorm2d(n_filters//4),
			nn.ReLU(),
		)

		self.branch3x3dbl = nn.Sequential(
			nn.Conv2d(n_filters, n_filters//4, 1, padding=0, bias=False),
			nn.BatchNorm2d(n_filters//4),
			nn.ReLU(),
			nn.Conv2d(n_filters//4, n_filters//4, 3, padding=1, bias=False),
			nn.BatchNorm2d(n_filters//4),
			nn.ReLU(),
			nn.Conv2d(n_filters//4, n_filters//4, 3, padding=1, bias=False),
			nn.BatchNorm2d(n_filters//4),
			nn.ReLU(),
		)

		self.branch_pool = nn.Sequential(
			nn.MaxPool2d(5, padding=2, stride=1) if pool == 'max' else nn.AvgPool2d(5, padding=2, stride=1),
			nn.Conv2d(n_filters, n_filters//4, 1, padding=0, bias=False),
			nn.BatchNorm2d(n_filters//4),
			nn.ReLU(),
		)

	def _forward(self, x):
		branch1x1 = self.branch1x1(x)
		branch5x5 = self.branch5x5(x)
		branch3x3dbl = self.branch3x3dbl(x)
		branch_pool = self.branch_pool(x)

		outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
		return outputs

	def forward(self, x):
		outputs = self._forward(x)
		return torch.cat(outputs, 1)

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
				nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False), nn.BatchNorm2d(n_filters), nn.ReLU(),
			)

			self.output_layers_PI = nn.Sequential(
				nn.Linear(n_filters*5*5, 128),
				nn.Linear(128, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Linear(n_filters*5*5, 32),
				nn.Linear(32, self.num_players)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Linear(n_filters*5*5, 32),
				nn.Linear(32, self.num_scdiffs*self.scdiff_size)
			)

		elif self.version == 51:
			n_filters = 64

			self.first_layer = nn.Conv2d(  2, n_filters, 3, padding=1, bias=False)
			self.trunk = nn.Sequential(
				Residualv1(n_filters),
				Residualv1(n_filters),
				Residualv1(n_filters),
			)

			self.output_layers_PI = nn.Sequential(
				nn.Linear(n_filters*5*5, 128),
				nn.Linear(128, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Linear(n_filters*5*5, 32),
				nn.Linear(32, self.num_players)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Linear(n_filters*5*5, 32),
				nn.Linear(32, self.num_scdiffs*self.scdiff_size)
			)

		elif self.version == 52:
			n_filters = 64

			self.first_layer = nn.Conv2d(  2, n_filters, 3, padding=1, bias=False)
			self.trunk = nn.Sequential(
				Residualv2(n_filters),
				Residualv2(n_filters),
				Residualv2(n_filters),
			)

			self.output_layers_PI = nn.Sequential(
				nn.Linear(n_filters*5*5, 128),
				nn.Linear(128, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Linear(n_filters*5*5, 32),
				nn.Linear(32, self.num_players)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Linear(n_filters*5*5, 32),
				nn.Linear(32, self.num_scdiffs*self.scdiff_size)
			)

		elif self.version == 53:
			n_filters = 64

			self.first_layer = nn.Conv2d(  2, n_filters, 3, padding=1, bias=False)
			self.trunk = nn.Sequential(
				SE_Residualv2(n_filters),
				SE_Residualv2(n_filters),
				SE_Residualv2(n_filters),
			)

			self.output_layers_PI = nn.Sequential(
				nn.Linear(n_filters*5*5, 128),
				nn.Linear(128, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Linear(n_filters*5*5, 32),
				nn.Linear(32, self.num_players)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Linear(n_filters*5*5, 32),
				nn.Linear(32, self.num_scdiffs*self.scdiff_size)
			)

		elif self.version == 54:
			n_filters = 64

			self.first_layer = nn.Conv2d(  2, n_filters, 3, padding=1, bias=False)
			self.trunk = nn.Sequential(
				SE_Residualv2(n_filters),
				SE_Residualv2(n_filters),
			)

			self.output_layers_PI = nn.Sequential(
				nn.Linear(n_filters*5*5, 128),
				nn.Linear(128, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Linear(n_filters*5*5, 32),
				nn.Linear(32, self.num_players)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Linear(n_filters*5*5, 32),
				nn.Linear(32, self.num_scdiffs*self.scdiff_size)
			)

		elif self.version == 55:
			n_filters = 64

			self.first_layer = nn.Conv2d(  2, n_filters, 3, padding=1, bias=False)
			self.trunk = nn.Sequential(
				SE_Residualv2(n_filters, pool='avg'),
				SE_Residualv2(n_filters, pool='avg'),
				SE_Residualv2(n_filters, pool='avg'),
			)

			self.output_layers_PI = nn.Sequential(
				nn.Linear(n_filters*5*5, 128),
				nn.Linear(128, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Linear(n_filters*5*5, 32),
				nn.Linear(32, self.num_players)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Linear(n_filters*5*5, 32),
				nn.Linear(32, self.num_scdiffs*self.scdiff_size)
			)
	
		elif self.version == 56:
			n_filters = 64

			self.first_layer = nn.Conv2d(  2, n_filters, 3, padding=1, bias=False)
			self.trunk = nn.Sequential(
				Residualv2(n_filters),
				Residualv2(n_filters),
				Residualv2(n_filters),
			)

			self.output_layers_PI = nn.Sequential(
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=False),
				nn.BatchNorm2d(n_filters),
				nn.ReLU(),
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=True),
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=False),
				nn.BatchNorm2d(n_filters),
				nn.ReLU(),
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=True),
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.num_players)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Conv2d(n_filters, n_filters//2, 1, padding=0, bias=True),
				nn.Flatten(1),
				nn.Linear(n_filters//2 *5*5, self.num_scdiffs*self.scdiff_size)
			)

		elif self.version == 57:
			n_filters = 64

			self.first_layer = nn.Conv2d(  2, n_filters, 3, padding=1, bias=False)
			self.trunk = nn.Sequential(
				SE_Residualv2(n_filters),
				SE_Residualv2(n_filters),
				SE_Residualv2(n_filters),
			)

			self.output_layers_PI = nn.Sequential(
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=False),
				nn.BatchNorm2d(n_filters),
				nn.ReLU(),
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=True),
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=False),
				nn.BatchNorm2d(n_filters),
				nn.ReLU(),
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=True),
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.num_players)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Conv2d(n_filters, n_filters//2, 1, padding=0, bias=True),
				nn.Flatten(1),
				nn.Linear(n_filters//2 *5*5, self.num_scdiffs*self.scdiff_size)
			)

		elif self.version == 58:
			n_filters = 96

			self.first_layer = nn.Conv2d(  2, n_filters, 3, padding=1, bias=False)
			self.trunk = nn.Sequential(
				InceptionA(n_filters),
				InceptionA(n_filters),
				InceptionA(n_filters),
			)

			self.output_layers_PI = nn.Sequential(
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=False),
				nn.BatchNorm2d(n_filters),
				nn.ReLU(),
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=True),
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=False),
				nn.BatchNorm2d(n_filters),
				nn.ReLU(),
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=True),
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.num_players)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Conv2d(n_filters, n_filters//2, 1, padding=0, bias=True),
				nn.Flatten(1),
				nn.Linear(n_filters//2 *5*5, self.num_scdiffs*self.scdiff_size)
			)

		elif self.version == 59:
			n_filters = 80

			self.first_layer = nn.Conv2d(  2, n_filters, 3, padding=1, bias=False)
			self.trunk = nn.Sequential(
				InceptionA(n_filters),
				InceptionA(n_filters),
				InceptionA(n_filters),
				InceptionA(n_filters),
			)

			self.output_layers_PI = nn.Sequential(
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=False),
				nn.BatchNorm2d(n_filters),
				nn.ReLU(),
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=True),
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=False),
				nn.BatchNorm2d(n_filters),
				nn.ReLU(),
				nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=True),
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.num_players)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Conv2d(n_filters, n_filters//2, 1, padding=0, bias=True),
				nn.Flatten(1),
				nn.Linear(n_filters//2 *5*5, self.num_scdiffs*self.scdiff_size)
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
		if self.version in [50, 51, 52, 53, 54, 55, 60]:
			x = input_data.transpose(-1, -2).view(-1, 3, 5, 5)
			x, data = x.split([2,1], dim=1)

			x = self.first_layer(x)
			x = self.trunk(x)
			x = torch.flatten(x, start_dim=1).unsqueeze(1)
			
			v = self.output_layers_V(x).squeeze(1)
			sdiff = self.output_layers_SDIFF(x).squeeze(1)
			pi = torch.where(valid_actions, self.output_layers_PI(x).squeeze(1), self.lowvalue)

		elif self.version in [56, 57, 58, 59]:
			x = input_data.transpose(-1, -2).view(-1, 3, 5, 5)
			x, data = x.split([2,1], dim=1)

			x = self.first_layer(x)
			x = self.trunk(x)
			
			v = self.output_layers_V(x)
			sdiff = self.output_layers_SDIFF(x)
			pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)


		return F.log_softmax(pi, dim=1), torch.tanh(v), F.log_softmax(sdiff.view(-1, self.num_scdiffs, self.scdiff_size).transpose(1,2), dim=1) # TODO
