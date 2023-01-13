import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.efficientnet import MBConvConfig, FusedMBConvConfig
from torchvision.models.mobilenetv3 import InvertedResidualConfig, InvertedResidual

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




def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SELayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.MaxPool2d(5)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None,
				 *, reduction=16):
		super(SEBasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes, 1)
		self.bn2 = nn.BatchNorm2d(planes)
		self.se = SELayer(planes, reduction)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.se(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class SEBottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None,
				 *, reduction=16):
		super(SEBottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.se = SELayer(planes * 4, reduction)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)
		out = self.se(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

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

		elif self.version in [52, 53, 54, 55, 56, 57, 60, 61, 65, 66]:
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
			elif self.version == 65: ## Configurable
				# Need to specify 'n_filters', 'expansion' (small, constant, progressive), 'depth'
				n_filters = self.args['n_filters']
				n_filters_first = self.args['n_filters']//4
				n_filters_begin = n_filters if self.args['expansion'] == 'constant' else n_filters_first
				n_filters_end = n_filters_first if self.args['expansion'] == 'small' else n_filters
				depth = self.args['depth'] - 2

				self.first_layer = nn.Conv2d(  2, n_filters_first, 3, padding=1, bias=False)
				confs  = [MBConvConfig(4, 3, 1, n_filters_first if i==0 else n_filters_begin, n_filters_begin, 1) for i in range(depth//2)]
				confs += [MBConvConfig(4, 3, 1, n_filters_begin if i==0 else n_filters_end  , n_filters_end  , 1) for i in range(depth//2)]
				confs += [MBConvConfig(4, 3, 1, n_filters_end                               , n_filters      , 1)]
				self.trunk = nn.Sequential(*[conf.block(conf, 0., nn.BatchNorm2d) for conf in confs])
			elif self.version == 66: ## Configurable
				# Need to specify 'n_filters', 'expansion' (small, constant, progressive), 'depth'
				n_filters = self.args['n_filters']
				n_filters_first = self.args['n_filters']//4
				n_filters_begin = n_filters if self.args['expansion'] == 'constant' else n_filters_first
				n_filters_end = n_filters_first if self.args['expansion'] == 'small' else n_filters
				n_exp_begin, n_exp_end = n_filters_begin*2, n_filters_end*3
				depth = self.args['depth'] - 2

				self.first_layer = nn.Conv2d(  2, n_filters_first, 3, padding=1, bias=False)
				confs  = [InvertedResidualConfig(n_filters_first if i==0 else n_filters_begin, 3, n_exp_begin, n_filters_begin, False, "RE", 1, 1, 1) for i in range(depth//2)]
				confs += [InvertedResidualConfig(n_filters_begin if i==0 else n_filters_end  , 3, n_exp_end  , n_filters_end  , True, "HS", 1, 1, 1) for i in range(depth//2)]
				confs += [InvertedResidualConfig(n_filters_end                               , 3, n_exp_end  , n_filters      , True, "HS", 1, 1, 1)]
				self.trunk = nn.Sequential(*[InvertedResidual(conf, nn.BatchNorm2d) for conf in confs])

			if self.version == 65:
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
			elif self.version == 66:
				head_depth = self.args['head_depth']
				n_exp_head = head_depth * 3
				confs_PI = [InvertedResidualConfig(n_filters, 3, n_exp_head, n_filters, True, "HS", 1, 1, 1) for i in range(head_depth)]
				head_PI = [InvertedResidual(conf, nn.BatchNorm2d) for conf in confs_PI] + [
					nn.Flatten(1),
					nn.Linear(n_filters *5*5, self.action_size),
					nn.ReLU(),
					nn.Linear(self.action_size, self.action_size)
				]
				self.output_layers_PI = nn.Sequential(*head_PI)

				confs_V = [InvertedResidualConfig(n_filters, 3, n_exp_head, n_filters, True, "HS", 1, 1, 1) for i in range(head_depth)]
				head_V = [InvertedResidual(conf, nn.BatchNorm2d) for conf in confs_V] + [
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

		elif self.version in [64, 52, 53, 54, 55, 56, 57, 60, 61, 65, 66]:
			x = input_data.transpose(-1, -2).view(-1, 3, 5, 5)
			x, data = x.split([2,1], dim=1)

			x = self.first_layer(x)
			x = self.trunk(x)
			
			v = self.output_layers_V(x)
			sdiff = self.output_layers_SDIFF(x)
			pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)

		return F.log_softmax(pi, dim=1), torch.tanh(v), F.log_softmax(sdiff.view(-1, self.num_scdiffs, self.scdiff_size).transpose(1,2), dim=1) # TODO
