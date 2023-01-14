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