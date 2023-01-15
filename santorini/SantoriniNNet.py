import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.efficientnet import MBConvConfig, FusedMBConvConfig
from torchvision.models.mobilenetv3 import InvertedResidualConfig, InvertedResidual
from .SantoriniNNet_blocks import Conv2dAndPartialMaxPool, SE_Residualv2, GARB, Global_Residual, Global_Head

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

		elif self.version == 24:
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
			self.dense1d_0 = nn.Sequential(
				nn.Linear(5*5, 5*5), nn.BatchNorm1d(1), nn.ReLU(),
				nn.Linear(5*5, 5*5), nn.BatchNorm1d(1), nn.ReLU(),
			)


			self.dense1d_1 = nn.Sequential(
				nn.Linear((128+1)*5*5, 512), nn.ReLU(),
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

		elif self.version in [52, 53, 54, 55, 56, 57, 60, 61, 65, 66, 67, 68]:
			n_filters = 128

			self.first_layer = nn.Conv2d(  2, n_filters, 3, padding=1, bias=False)
			if self.version == 64:
				self.trunk = nn.Sequential(*[SE_Residualv2(n_filters) for _ in range (5)])
			elif self.version == 52:
				self.trunk = nn.Sequential(*[BasicBlock(n_filters, n_filters) for _ in range (4)])
			elif self.version == 54 or self.version == 57: # bottleneck style
				n_filters = 512
				self.first_layer = nn.Conv2d(  2, 256, 3, padding=1, bias=False)
				self.trunk = nn.Sequential(*[Bottleneck(256, 256//4) for _ in range (3)])
			elif self.version == 60:
				downsample_trunk = nn.Sequential(
					nn.Conv2d(n_filters, n_filters//2, 1, bias=False),
					nn.BatchNorm2d(n_filters//2)
				)
				trunk  = [Bottleneck(n_filters, n_filters//4, groups=32, base_width=8) for _ in range (1)]
				trunk += [Bottleneck(n_filters, n_filters//8, groups=32, base_width=8, downsample=downsample_trunk)]
				self.trunk = nn.Sequential(*trunk)
			elif self.version == 61:
				self.first_layer = nn.Conv2d(  2, 16, 3, padding=1, bias=False)
				confs = [
					# expand_ratio, kernel, stride, input_channels, out_channels, arg_pas_utile
					FusedMBConvConfig(1, 3, 1, 16, 16, 1),
					FusedMBConvConfig(4, 3, 1, 16, 32, 1),
					MBConvConfig(     4, 3, 1, 32, 32, 1),
					MBConvConfig(     4, 3, 1, 32, 64, 1),
					MBConvConfig(     6, 3, 1, 64, 64, 1),
					MBConvConfig(     6, 3, 1, 64, 64, 1),
				]
				self.trunk = nn.Sequential(*[conf.block(conf, 0., nn.BatchNorm2d) for conf in confs])

			##### Configurable EfficientNet #####
			elif self.version == 65: 
				expansion_str = ['small', 'constant', 'progressive'][self.args['expansion']]
				# Need to specify 'n_filters', 'expansion' (small, constant, progressive), 'depth'
				n_filters = self.args['n_filters']
				n_filters_first = self.args['n_filters']//4
				n_filters_begin = n_filters if expansion_str == 'constant' else n_filters_first
				n_filters_end = n_filters_first if expansion_str == 'small' else n_filters
				depth = self.args['depth'] - 2

				self.first_layer = nn.Conv2d(  2, n_filters_first, 3, padding=1, bias=False)
				confs  = [MBConvConfig(4, 3, 1, n_filters_first if i==0 else n_filters_begin, n_filters_begin, 1) for i in range(depth//2)]
				confs += [MBConvConfig(4, 3, 1, n_filters_begin if i==0 else n_filters_end  , n_filters_end  , 1) for i in range(depth//2)]
				confs += [MBConvConfig(4, 3, 1, n_filters_end                               , n_filters      , 1)]
				self.trunk = nn.Sequential(*[conf.block(conf, 0., nn.BatchNorm2d) for conf in confs])

			##### Configurable MobileNet #####
			elif self.version == 66:
				expansion_str = ['small', 'constant', 'progressive'][self.args['expansion']]
				# Need to specify 'n_filters', 'expansion' (small, constant, progressive), 'depth'
				n_filters = self.args['n_filters']
				n_filters_first = self.args['n_filters']//4
				n_filters_begin = n_filters if expansion_str == 'constant' else n_filters_first
				n_filters_end = n_filters_first if expansion_str == 'small' else n_filters
				n_exp_begin, n_exp_end = n_filters_begin*2, n_filters_end*3
				depth = self.args['depth'] - 2

				def inverted_residual(input_ch, expanded_ch, out_ch, use_se, activation, kernel=3, stride=1, dilation=1, width_mult=1.):
					return InvertedResidual(InvertedResidualConfig(input_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation, width_mult), nn.BatchNorm2d)
				self.first_layer = nn.Conv2d(  2, n_filters_first, 3, padding=1, bias=False)
				confs  = [inverted_residual(n_filters_first if i==0 else n_filters_begin, n_exp_begin, n_filters_begin, False, "RE") for i in range(depth//2)]
				confs += [inverted_residual(n_filters_begin if i==0 else n_filters_end  , n_exp_end  , n_filters_end  , True , "HS") for i in range(depth//2)]
				confs += [inverted_residual(n_filters_end                               , n_exp_end  , n_filters      , True , "HS")]
				self.trunk = nn.Sequential(*confs)

			##### Configurable GARB (NoGoZero+) #####
			elif self.version == 67:
				# Need to specify 'n_filters', 'global_every_n', 'depth'
				n_filters = self.args['n_filters']
				global_every_n = self.args['expansion']
				depth = self.args['depth'] - 1

				self.first_layer = nn.Conv2d(  2, n_filters, 3, padding=1, bias=False)
				trunk = [GARB(n_filters) if i % global_every_n == 0 else BasicBlock(n_filters, n_filters) for i in range(depth)]
				self.trunk = nn.Sequential(*trunk)

			##### Configurable KataGo #####
			elif self.version == 68:
				# Need to specify 'n_filters', 'global_every_n', 'depth'
				n_filters = self.args['n_filters']
				global_every_n = self.args['expansion']
				depth = self.args['depth'] - 1

				self.first_layer = nn.Conv2d(  2, n_filters, 3, padding=1, bias=False)
				trunk = [Global_Residual(n_filters) if i % global_every_n == 0 else BasicBlock(n_filters, n_filters) for i in range(depth)]
				self.trunk = nn.Sequential(*trunk)

			# Head
			if self.version == 65: ##### Configurable EfficientNet #####
				head_depth = self.args['head_depth']
				confs_PI = [MBConvConfig(4, 3, 1, n_filters, n_filters, 1) for i in range(head_depth)]
				head_PI = [conf.block(conf, 0., nn.BatchNorm2d) for conf in confs_PI] + [
					nn.Flatten(1),
					nn.Linear(n_filters *5*5, self.action_size),
					nn.ReLU(),
					nn.Linear(self.action_size, self.action_size)
				]
				self.output_layers_PI = nn.Sequential(*head_PI)

				confs_V = [MBConvConfig(4, 3, 1, n_filters, n_filters, 1) for i in range(head_depth)]
				head_V = [conf.block(conf, 0., nn.BatchNorm2d) for conf in confs_V] + [
					nn.Flatten(1),
					nn.Linear(n_filters *5*5, self.num_players),
					nn.ReLU(),
					nn.Linear(self.num_players, self.num_players)
				]
				self.output_layers_V = nn.Sequential(*head_V)

				self.output_layers_SDIFF = nn.Sequential(
					nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=True),
					nn.Flatten(1),
					nn.Linear(n_filters *5*5, self.num_scdiffs*self.scdiff_size)
				)
			elif self.version == 66: ##### Configurable MobileNet #####
				head_depth = self.args['head_depth']
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
				# self.output_layers_PI = Global_Head(n_filters, n_filters, self.action_size, self.action_size)
				# self.output_layers_V = Global_Head(n_filters, n_filters, self.num_players, self.num_players)

				self.output_layers_SDIFF = nn.Sequential(
					nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=True),
					nn.Flatten(1),
					nn.Linear(n_filters *5*5, self.num_scdiffs*self.scdiff_size)
				)
			elif self.version == 67: ##### Configurable GARB (NoGoZero+) #####
				head_depth = self.args['head_depth']
				head_PI = [GARB(n_filters) if i == head_depth//2 else BasicBlock(n_filters, n_filters) for i in range(head_depth)] + [
					nn.Flatten(1),
					nn.Linear(n_filters *5*5, self.action_size),
					nn.ReLU(),
					nn.Linear(self.action_size, self.action_size)
				]
				self.output_layers_PI = nn.Sequential(*head_PI)

				head_V = [GARB(n_filters) if i == head_depth//2 else BasicBlock(n_filters, n_filters) for i in range(head_depth)] + [
					nn.Flatten(1),
					nn.Linear(n_filters *5*5, self.num_players),
					nn.ReLU(),
					nn.Linear(self.num_players, self.num_players)
				]
				self.output_layers_V = nn.Sequential(*head_V)

				self.output_layers_SDIFF = nn.Sequential(
					nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=True),
					nn.Flatten(1),
					nn.Linear(n_filters *5*5, self.num_scdiffs*self.scdiff_size)
				)
			elif self.version == 68: ##### Configurable KataGo #####
				self.output_layers_PI = Global_Head(n_filters, n_filters, self.action_size, self.action_size)
				self.output_layers_V = Global_Head(n_filters, n_filters, self.num_players, self.num_players)
				self.output_layers_SDIFF = nn.Sequential(
					nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=True),
					nn.Flatten(1),
					nn.Linear(n_filters *5*5, self.num_scdiffs*self.scdiff_size)
				)
			else:
				self.output_layers_PI = nn.Sequential(
					Bottleneck(n_filters, n_filters//4),
					Bottleneck(n_filters, n_filters//4),
					nn.Flatten(1),
					nn.Linear(n_filters *5*5, self.action_size),
					nn.ReLU(),
					nn.Linear(self.action_size, self.action_size)
				)

				self.output_layers_V = nn.Sequential(
					Bottleneck(n_filters, n_filters//4),
					Bottleneck(n_filters, n_filters//4),
					nn.Flatten(1),
					nn.Linear(n_filters *5*5, self.num_players),
					nn.ReLU(),
					nn.Linear(self.num_players, self.num_players)
				)

				self.output_layers_SDIFF = nn.Sequential(
					nn.Conv2d(n_filters, n_filters, 1, padding=0, bias=True),
					nn.Flatten(1),
					nn.Linear(n_filters *5*5, self.num_scdiffs*self.scdiff_size)
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
		if self.version in [20, 21, 22, 23, 24, 25]:
			x = input_data.transpose(-1, -2).view(-1, 3, 5, 5)
			x, data = x.split([2,1], dim=1)

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
			
			v = self.output_layers_V(x).squeeze(1)
			sdiff = self.output_layers_SDIFF(x).squeeze(1)
			pi = torch.where(valid_actions, self.output_layers_PI(x).squeeze(1), self.lowvalue)

		elif self.version in [64, 52, 53, 54, 55, 56, 57, 60, 61, 65, 66, 67, 68]:
			x = input_data.transpose(-1, -2).view(-1, 3, 5, 5)
			x, data = x.split([2,1], dim=1)

			x = self.first_layer(x)
			x = self.trunk(x)
			
			v = self.output_layers_V(x)
			sdiff = self.output_layers_SDIFF(x)
			pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)

		return F.log_softmax(pi, dim=1), torch.tanh(v), F.log_softmax(sdiff.view(-1, self.num_scdiffs, self.scdiff_size).transpose(1,2), dim=1) # TODO
