import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _make_divisible
from torchvision.models.mobilenetv3 import InvertedResidualConfig, InvertedResidual

from .AkropolisConstants import N_COLORS, CITY_SIZE, CONSTR_SITE_SIZE, CODES_LIST

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

class AkropolisNNet(nn.Module):
	def __init__(self, game, args):
		# game params
		# self.dim1, self.dim2, self.dim3 = game.getBoardSize()
		# self.dim1d_per_pl = N_COLORS
		# self.dim3d_per_pl = (CITY_SIZE, CITY_SIZE, 2)
	
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.args = args
		self.version = args['nn_version']
		super(AkropolisNNet, self).__init__()

		def inverted_residual(input_ch, expanded_ch, out_ch, use_se, activation, kernel=3, stride=1, dilation=1, width_mult=1):
			return InvertedResidual(InvertedResidualConfig(input_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation, width_mult), nn.BatchNorm2d)

		def filters_for_boards(input_ch, expanded_ch, out_ch, depth):
			filters  = [nn.Conv2d(input_ch, input_ch, kernel_size=3, padding=1, bias=False)]
			filters += [inverted_residual(input_ch, expanded_ch, input_ch, False, "RE") for i in range(depth//2)]
			filters += [inverted_residual(input_ch, expanded_ch, out_ch, False, "RE")]
			filters += [inverted_residual(out_ch, expanded_ch, out_ch, True, "HS") for i in range(depth//2)]
			filters += [nn.Flatten()]
			return nn.Sequential(*filters)

		if self.version == 1: # Ultra simple NN
			num_filters = 100
			self.trunk_1d = nn.Linear(3*self.num_players*N_COLORS+3*CONSTR_SITE_SIZE, num_filters)
			self.final_layers_V = nn.Linear(num_filters, 1)
			self.final_layers_PI = nn.Sequential(nn.Linear(num_filters, self.action_size), nn.Linear(self.action_size, self.action_size))
		elif self.version == 8: # Simple version using Embedding
			D = 5
			T = 16
			G = 32
			S = 8
			self.embed = nn.Embedding(num_embeddings=len(CODES_LIST), embedding_dim=D)
			self.conv1d_constr = nn.Conv1d(D, T, kernel_size=3, stride=1, padding=0)
			self.dense_scores = nn.Linear(N_COLORS*3*self.num_players, G)
			self.dense_globs = nn.Linear(2, S)
			self.fused_to_1d = nn.MaxPool1d(T+G+S)
			self.final_layers_V = nn.Sequential(
				nn.Linear(CONSTR_SITE_SIZE, 1)
			)
			self.final_layers_PI = nn.Sequential(
				nn.Linear(CONSTR_SITE_SIZE, self.action_size)
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
		# Switch from NHWC to NCHW
		x = input_data.reshape(-1, CITY_SIZE, CITY_SIZE, 3*self.num_players+2).permute(0, 3, 1, 2)
		# Split data
		split_input_data = x.split([self.num_players, self.num_players, self.num_players, 1 ,1], dim=1)
		boards_descr, boards_height, _, per_pl_data, global_data = split_input_data
		scores_data = per_pl_data.squeeze(1)[:, :3*self.num_players, :N_COLORS]
		constrs_site = global_data.squeeze(1)[:, :CONSTR_SITE_SIZE, :3]
		globals_data = global_data.squeeze(1)[:, CONSTR_SITE_SIZE+1, :2]
		# x.shape = Nx8x12x12
		# boards_descr.shape = boards_height.shape = Nx2x12x12
		# per_pl_data.shape = global_data.shape = Nx1x12x12
		# scores_data.shape = Nx6x5
		# constrs_site.shape = Nx3x3

		if self.version in [1]:
			x_1d = torch.cat([scores_data.flatten(1), constrs_site.flatten(1)], dim=1) # x_1d.shape = Nx39
			x_1d = F.dropout(self.trunk_1d(x_1d), p=self.args['dropout'], training=self.training) # x_1d.shape = Nx100
			v = self.final_layers_V(x_1d)
			pi = torch.where(valid_actions, self.final_layers_PI(x_1d), self.lowvalue)
		elif self.version in [8]:
			# Convert constrs_site to embeddings (need to clamp when input is random)
			constrs_long = constrs_site.clamp(min=0., max=len(CODES_LIST)-1).long()
			constrs_embed = self.embed(constrs_long) # N,CS,3,D
			constrs_3d = constrs_embed.flatten(start_dim=0, end_dim=1).permute(0, 2, 1) # N*CS, D, 3
			constrs_3d = F.dropout(self.conv1d_constr(constrs_3d), p=self.args['dropout'], training=self.training) # N*CS, T
			constrs_3d = constrs_3d.squeeze(-1).view(constrs_embed.shape[0], constrs_embed.shape[1], -1) # N, CS, T

			s1 = F.dropout(self.dense_scores(scores_data.flatten(1)), p=self.args['dropout'], training=self.training) # N,3*NUM_PLAYERS,N_COLORS -> N,G
			scores_3d = s1.unsqueeze(1).expand(-1, CONSTR_SITE_SIZE, -1) # N, CS, G

			g1 = F.dropout(self.dense_globs(globals_data), p=self.args['dropout'], training=self.training) # 2 -> N,S
			glob_3d = g1.unsqueeze(1).expand(-1, CONSTR_SITE_SIZE, -1) # N, CS, S

			fused = torch.cat([constrs_3d, scores_3d, glob_3d], dim=-1) # N,CS,T+G+S
			fused_1d = self.fused_to_1d(fused).squeeze(-1)
			v = self.final_layers_V(fused_1d)
			pi = torch.where(valid_actions, self.final_layers_PI(fused_1d), self.lowvalue)
		else:
			raise Exception(f'Unsupported NN version {self.version}')

		return F.log_softmax(pi, dim=1), torch.tanh(v)

