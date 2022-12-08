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

class Global_Residual(nn.Module):
	def __init__(self, n_filters, pool_factor=3):
		super().__init__()
		self.n_filters = n_filters
		self.global_filters = n_filters // pool_factor
		remaining_filters = n_filters - self.global_filters

		self.conv_path = nn.Sequential(
			nn.BatchNorm2d(n_filters),
			nn.ReLU(),
			nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=True),
		)

		self.global_path = nn.Sequential(
			nn.BatchNorm2d(self.global_filters),
			nn.ReLU(),
		)
		self.global_max = nn.MaxPool2d(5)
		self.global_avg = nn.AvgPool2d(5)
		self.global_dense = nn.Linear(2*self.global_filters, remaining_filters)

		self.final_path = nn.Sequential(
			nn.BatchNorm2d(remaining_filters),
			nn.ReLU(),
			nn.Conv2d(remaining_filters, n_filters, 3, padding=1, bias=True),
		)

	def forward(self, x):
		y = self.conv_path(x)

		split_y = y.split([self.global_filters, self.n_filters - self.global_filters], dim=1)
		attention = self.global_path(split_y[0])
		attention_max, attention_avg = self.global_max(attention), self.global_avg(attention)
		attention = self.global_dense(torch.cat([attention_max, attention_avg], 1).flatten(1))
		y = split_y[1] + attention.unsqueeze(-1).unsqueeze(-1)
		y = self.final_path(y)

		x = x + y
		return x

class Global_Head(nn.Module):
	def __init__(self, n_filters, p_filters, d_filters, out_filters, pool_factor=3):
		super().__init__()
		self.path_2d = nn.Sequential(
			Global_Residual(n_filters, pool_factor),
			nn.Conv2d(n_filters, p_filters, 1, padding=0, bias=True),
		)
		self.path_1d = nn.Sequential(
			nn.Linear(p_filters*5*5, d_filters  ), nn.BatchNorm1d(d_filters), nn.ReLU(),
			nn.Linear(d_filters    , d_filters  ), nn.BatchNorm1d(d_filters), nn.ReLU(),
			nn.Linear(d_filters    , out_filters),
		)

	def forward(self, x):
		x = self.path_2d(x)
		x = self.path_1d(x.flatten(1))
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

# Assume 4-dim tensor input N,C,H,W
#
# Input            Output
#  C1    --↘  ↗--> Conv2D(C1 .. C3)
#  C2    ---==---> Conv2D(C1 .. C3)			Regular Conv2D
#  C3    --↗  ↘--> Conv2D(C1 .. C3)
#  C4    --------> MaxPool2D(C4) 			2 maxplanar, using kernel 3 on each plan
#  C5    --------> MaxPool2D(C5)
#  C6    -↘
#  C7    --------> Max(C6,C7,C8)
#  C8    -↗                                 2 groups of maxchannels, with 3 channels each
#  C9    -↘
#  C10   --------> Max(C9,C10,C11)
#  C11   -↗
class Conv2dAndPartialMaxPool(nn.Module):
	def __init__(self, input_length, output_length, kernel_conv=3, nb_channel_maxplanar=4, kernel_maxplanar=3, nb_groups_maxchannel=4, kernel_maxchannel=4, batchnorm=True):
		super().__init__()
		self.params_maxpool_planar  = (nb_channel_maxplanar, kernel_maxplanar)
		self.params_maxpool_channel = (nb_groups_maxchannel, kernel_maxchannel)

		self.conv_input = input_length - nb_channel_maxplanar - nb_groups_maxchannel*kernel_maxchannel
		self.conv_output = output_length - nb_channel_maxplanar - nb_groups_maxchannel
		self.conv_part = nn.Sequential(
			nn.Conv2d(self.conv_input, self.conv_output, kernel_conv, padding=kernel_conv//2),
			nn.BatchNorm2d(self.conv_output) if batchnorm else nn.Identity()
		)
		self.maxplanar  = nn.MaxPool2d(kernel_maxplanar, stride=1, padding=kernel_maxplanar//2)
		self.maxchannel = nn.MaxPool2d((kernel_maxchannel,1))

	def forward(self, x):
		groups_for_gpool = x.split([1] * self.params_maxpool_planar[0] + [self.params_maxpool_channel[1]] * self.params_maxpool_channel[0] + [self.conv_input], 1)
		# Max over plan, can use MaxPool2d directly
		maxplanar_results  = [ self.maxplanar(y) for y in groups_for_gpool[:self.params_maxpool_planar[0]] ]
		# Max over channels, need to permute dimensions before using MaxPool2d
		maxchannel_results = [ self.maxchannel(y.permute(0, 2, 1, 3)).permute(0, 2, 1, 3) for y in groups_for_gpool[self.params_maxpool_planar[0]:-1] ]
		conv_result = F.relu(self.conv_part(groups_for_gpool[-1]))

		x = torch.cat(maxplanar_results + maxchannel_results + [conv_result], 1)
		return x

class GARB(nn.Module):
	def __init__(self, n_filters):
		super().__init__()
		self.n_filters = n_filters
		global_filters = n_filters // 3
		conv_filters = n_filters - global_filters

		self.global_path = nn.Sequential(
			nn.BatchNorm2d(global_filters),
			nn.ReLU(),
			nn.Conv2d(global_filters, global_filters, 3, padding=1, bias=False),
			nn.BatchNorm2d(global_filters),
			nn.ReLU(),
		)
		self.global_max = nn.MaxPool2d(5)
		self.global_avg = nn.AvgPool2d(5)
		self.global_dense = nn.Linear(2*global_filters, conv_filters)

		self.conv_path = nn.Sequential(
			nn.BatchNorm2d(conv_filters),
			nn.ReLU(),
			nn.Conv2d(conv_filters, conv_filters, 3, padding=1, bias=True),
		)

		self.final_path = nn.Sequential(
			nn.BatchNorm2d(conv_filters),
			nn.ReLU(),
			nn.Conv2d(conv_filters, n_filters, 3, padding=1, bias=True),
		)

	def forward(self, x):
		split_input = x.split([self.n_filters//3, self.n_filters - self.n_filters//3], dim=1)

		attention = self.global_path(split_input[0])
		attention_max, attention_avg = self.global_max(attention), self.global_avg(attention)
		attention = self.global_dense(torch.cat([attention_max, attention_avg], 1).flatten(1))

		y = self.conv_path(split_input[1])
		y = y + attention.unsqueeze(-1).unsqueeze(-1)

		y = self.final_path(y)
		x = x + y
		return x

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
		
		elif self.version in [23, 24, 25, 26, 27, 30, 31, 32]:
			self.conv2d_1 = nn.Sequential(
				nn.Conv2d(  2, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
				nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
			)
			self.conv2d_2 = nn.Sequential(
				nn.Conv2d(128, 128, 3, padding=1)                     , nn.ReLU(),
				nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
			)
			if self.version < 30:
				self.partialgpool_1 = Conv2dAndPartialMaxPool(128, 128, kernel_conv=3, nb_channel_maxplanar=4, kernel_maxplanar=3, nb_groups_maxchannel=2, kernel_maxchannel=4)
			elif self.version == 30:
				self.partialgpool_1 = Conv2dAndPartialMaxPool(128, 128, kernel_conv=3, nb_channel_maxplanar=0, kernel_maxplanar=3, nb_groups_maxchannel=2, kernel_maxchannel=4)
			elif self.version == 31:
				self.partialgpool_1 = Conv2dAndPartialMaxPool(128, 128, kernel_conv=3, nb_channel_maxplanar=4, kernel_maxplanar=3, nb_groups_maxchannel=0, kernel_maxchannel=4)
			elif self.version == 32:
				self.partialgpool_1 = Conv2dAndPartialMaxPool(128, 128, kernel_conv=3, nb_channel_maxplanar=0, kernel_maxplanar=3, nb_groups_maxchannel=0, kernel_maxchannel=4)

			self.conv2d_3 = nn.Sequential(
				nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
				nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
			)
			self.dense1d_0 = nn.Sequential(
				nn.Linear(5*5, 5*5), nn.BatchNorm1d(1), nn.ReLU(),
				nn.Linear(5*5, 5*5), nn.BatchNorm1d(1), nn.ReLU(),
			)

			if self.version == 24:
				self.dense1d_1 = nn.Sequential(
					nn.Linear((128+1)*5*5, 512), nn.ReLU(),
				)
			else:
				self.dense1d_1 = nn.Sequential(
					nn.Linear(128*5*5, 512), nn.ReLU(),
				)
			self.dense1d_2 = nn.Sequential(
				nn.Linear(512, 512), nn.BatchNorm1d(1), nn.ReLU(),
				nn.Linear(512, 256), nn.BatchNorm1d(1), nn.ReLU(),
			)

			self.output_layers_PI = nn.Sequential(
				nn.Linear(256, 256),
				nn.Linear(256, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Linear(256, 256),
				nn.Linear(256, self.num_players)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Linear(256, 256),
				nn.Linear(256, self.num_scdiffs*self.scdiff_size)
			)

		elif self.version in [28, 29, 33, 34, 35]:
			if self.version == 28:
				conv = Residualv2 
				self.partialgpool_1 = Conv2dAndPartialMaxPool(128, 128, kernel_conv=3, nb_channel_maxplanar=4, kernel_maxplanar=3, nb_groups_maxchannel=2, kernel_maxchannel=4)
			elif self.version == 29:
				conv = Residualv2 
				self.partialgpool_1 = GARB(128)
			elif self.version == 33:
				conv = SE_Residualv2
				self.partialgpool_1 = Conv2dAndPartialMaxPool(128, 128, kernel_conv=3, nb_channel_maxplanar=4, kernel_maxplanar=3, nb_groups_maxchannel=2, kernel_maxchannel=4)
			elif self.version == 34:
				conv = Residualv2 
				self.partialgpool_1 = Global_Residual(128)
			elif self.version == 35:
				conv = Residualv2 
				self.partialgpool_1 = Global_Residual(128, pool_factor=4)

			self.conv2d_1 = nn.Conv2d(  2, 128, 3, padding=1)
			self.conv2d_2 = conv(128)
			self.conv2d_3 = nn.Sequential(conv(128), conv(128))

			self.dense1d_0 = nn.Sequential(
				nn.Linear(5*5, 5*5), nn.BatchNorm1d(1), nn.ReLU(),
				nn.Linear(5*5, 5*5), nn.BatchNorm1d(1), nn.ReLU(),
			)


			self.dense1d_1 = nn.Sequential(
				nn.Linear(128*5*5, 512), nn.ReLU(),
			)
			self.dense1d_2 = nn.Sequential(
				nn.Linear(512, 512), nn.BatchNorm1d(1), nn.ReLU(),
				nn.Linear(512, 256), nn.BatchNorm1d(1), nn.ReLU(),
			)

			self.output_layers_PI = nn.Sequential(
				nn.Linear(256, 256),
				nn.Linear(256, self.action_size)
			)

			self.output_layers_V = nn.Sequential(
				nn.Linear(256, 256),
				nn.Linear(256, self.num_players)
			)

			self.output_layers_SDIFF = nn.Sequential(
				nn.Linear(256, 256),
				nn.Linear(256, self.num_scdiffs*self.scdiff_size)
			)

		elif self.version in [36]:
			self.conv2d_1 = nn.Sequential(
				nn.Conv2d(  2, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
				nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
			)
			self.conv2d_2 = nn.Sequential(
				nn.Conv2d(128, 128, 3, padding=1)                     , nn.ReLU(),
				nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
			)
			
			self.partialgpool_1 = Conv2dAndPartialMaxPool(128, 128, kernel_conv=3, nb_channel_maxplanar=4, kernel_maxplanar=3, nb_groups_maxchannel=2, kernel_maxchannel=4)

			self.conv2d_3 = nn.Sequential(
				nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
				nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
			)

			self.output_layers_PI = Global_Head(128, 128, 256, self.action_size)
			self.output_layers_V = Global_Head(128, 128, 128, self.num_players)
			self.output_layers_SDIFF = Global_Head(128, 64, 64, self.num_scdiffs*self.scdiff_size)


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
			n_filters = 64

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

		elif self.version in [60, 61, 62, 63, 64, 65]:
			if self.version in [60, 61, 62, 64, 65]:
				n_filters = 128
			elif self.version in [63]:
				n_filters = 64

			self.first_layer = nn.Conv2d(  2, n_filters, 3, padding=1, bias=False)
			if self.version == 60:
				self.trunk = nn.Sequential(
					nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False), nn.BatchNorm2d(n_filters), nn.ReLU(),
					nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False), nn.BatchNorm2d(n_filters), nn.ReLU(),
					nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False), nn.BatchNorm2d(n_filters), nn.ReLU(),
					nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False), nn.BatchNorm2d(n_filters), nn.ReLU(),
					nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False), nn.BatchNorm2d(n_filters), nn.ReLU(),
					nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False), nn.BatchNorm2d(n_filters), nn.ReLU(),
					nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False), nn.BatchNorm2d(n_filters), nn.ReLU(),
				)
			elif self.version in [61, 62]:
				self.trunk = nn.Sequential(
					Residualv2(n_filters),
					Residualv2(n_filters),
					Residualv2(n_filters),
					Residualv2(n_filters),
				)
			elif self.version in [63]:
				self.trunk = nn.Sequential(
					Residualv2(n_filters),
					GARB(n_filters),
					Residualv2(n_filters),
					GARB(n_filters),
				)
			elif self.version in [64, 65]:
				self.trunk = nn.Sequential(
					SE_Residualv2(n_filters),
					SE_Residualv2(n_filters),
					SE_Residualv2(n_filters),
					SE_Residualv2(n_filters),
					SE_Residualv2(n_filters),
					SE_Residualv2(n_filters),
					SE_Residualv2(n_filters),
					SE_Residualv2(n_filters),
					SE_Residualv2(n_filters),
					SE_Residualv2(n_filters),
				)

			if self.version in [60, 61]:
				self.output_layers_PI = nn.Sequential(
					nn.Linear(n_filters*5*5, 256),
					nn.Linear(256, self.action_size)
				)
				self.output_layers_V = nn.Sequential(
					nn.Linear(n_filters*5*5, 32),
					nn.Linear(32, self.num_players)
				)
				self.output_layers_SDIFF = nn.Sequential(
					nn.Linear(n_filters*5*5, 32),
					nn.Linear(32, self.num_scdiffs*self.scdiff_size)
				)
			elif self.version in [63]:
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
			elif self.version in [62, 64]:
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
			elif self.version in [65]:
				self.output_layers_PI = Global_Head(128, 128, 256, self.action_size)
				self.output_layers_V = Global_Head(128, 128, 128, self.num_players)
				self.output_layers_SDIFF = Global_Head(128, 64, 64, self.num_scdiffs*self.scdiff_size)

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
		if self.version in [50, 51, 52, 53, 54, 55, 60, 61, 63]:
			x = input_data.transpose(-1, -2).view(-1, 3, 5, 5)
			x, data = x.split([2,1], dim=1)

			x = self.first_layer(x)
			x = self.trunk(x)
			x = torch.flatten(x, start_dim=1).unsqueeze(1)
			
			v = self.output_layers_V(x).squeeze(1)
			sdiff = self.output_layers_SDIFF(x).squeeze(1)
			pi = torch.where(valid_actions, self.output_layers_PI(x).squeeze(1), self.lowvalue)

		elif self.version in [56, 57, 58, 59, 62, 64, 65]:
			x = input_data.transpose(-1, -2).view(-1, 3, 5, 5)
			x, data = x.split([2,1], dim=1)

			x = self.first_layer(x)
			x = self.trunk(x)
			
			v = self.output_layers_V(x)
			sdiff = self.output_layers_SDIFF(x)
			pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)

		elif self.version in [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
			x = input_data.transpose(-1, -2).view(-1, 3, 5, 5)
			x, data = x.split([2,1], dim=1)

			if self.version == 23:
				x = F.dropout(self.conv2d_1(x)      , p=self.args['dropout'], training=self.training)
				x = F.dropout(self.conv2d_2(x)      , p=self.args['dropout'], training=self.training)
				x = F.dropout(self.partialgpool_1(x), p=self.args['dropout'], training=self.training)
				x = F.dropout(self.conv2d_3(x)      , p=self.args['dropout'], training=self.training)
				x = torch.flatten(x, start_dim=1).unsqueeze(1)
				x = F.dropout(self.dense1d_1(x)     , p=self.args['dropout'], training=self.training)
				x = F.dropout(self.dense1d_2(x)     , p=self.args['dropout'], training=self.training)
			elif self.version == 24:
				x = F.dropout(self.conv2d_1(x)      , p=self.args['dropout'], training=self.training)
				x = F.dropout(self.conv2d_2(x)      , p=self.args['dropout'], training=self.training)
				x = F.dropout(self.partialgpool_1(x), p=self.args['dropout'], training=self.training)
				x = F.dropout(self.conv2d_3(x)      , p=self.args['dropout'], training=self.training)
				data = torch.flatten(data, start_dim=2)
				data = F.dropout(self.dense1d_0(data), p=self.args['dropout'], training=self.training)
				x = torch.flatten(x, start_dim=1).unsqueeze(1)
				x = torch.cat([x, data], dim=-1)
				x = F.dropout(self.dense1d_1(x)     , p=self.args['dropout'], training=self.training)
				x = F.dropout(self.dense1d_2(x)     , p=self.args['dropout'], training=self.training)
			elif self.version in [25, 28, 29, 30, 31, 32, 33, 34, 35]:
				x = self.conv2d_1(x)
				x = self.conv2d_2(x)
				x = self.partialgpool_1(x)
				x = self.conv2d_3(x)
				x = torch.flatten(x, start_dim=1).unsqueeze(1)
				x = self.dense1d_1(x)
				x = self.dense1d_2(x)
			elif self.version == 26:
				x = F.dropout2d(self.conv2d_1(x)      , p=self.args['dropout'], training=self.training)
				x = F.dropout2d(self.conv2d_2(x)      , p=self.args['dropout'], training=self.training)
				x = F.dropout2d(self.partialgpool_1(x), p=self.args['dropout'], training=self.training)
				x = F.dropout2d(self.conv2d_3(x)      , p=self.args['dropout'], training=self.training)
				x = torch.flatten(x, start_dim=1).unsqueeze(1)
				x = F.dropout2d(self.dense1d_1(x)     , p=self.args['dropout'], training=self.training)
				x = F.dropout2d(self.dense1d_2(x)     , p=self.args['dropout'], training=self.training)
			elif self.version == 27:
				x = self.conv2d_1(x)
				x = self.conv2d_2(x)
				x = self.partialgpool_1(x)
				x = self.conv2d_3(x)
				x = torch.flatten(x, start_dim=1).unsqueeze(1)
				x = self.dense1d_1(x)
				x = F.dropout2d(self.dense1d_2(x)     , p=self.args['dropout'], training=self.training)
			else:
				raise Exception(f'Warning, unknown NN version {self.version}')

			v = self.output_layers_V(x).squeeze(1)
			sdiff = self.output_layers_SDIFF(x).squeeze(1)
			pi = torch.where(valid_actions, self.output_layers_PI(x).squeeze(1), self.lowvalue)

		elif self.version in [36]:
			x = input_data.transpose(-1, -2).view(-1, 3, 5, 5)
			x, data = x.split([2,1], dim=1)

			x = self.conv2d_1(x)
			x = self.conv2d_2(x)
			x = self.partialgpool_1(x)
			x = self.conv2d_3(x)

			v = self.output_layers_V(x).squeeze(1)
			sdiff = self.output_layers_SDIFF(x).squeeze(1)
			pi = torch.where(valid_actions, self.output_layers_PI(x).squeeze(1), self.lowvalue)

		return F.log_softmax(pi, dim=1), torch.tanh(v), F.log_softmax(sdiff.view(-1, self.num_scdiffs, self.scdiff_size).transpose(1,2), dim=1) # TODO
