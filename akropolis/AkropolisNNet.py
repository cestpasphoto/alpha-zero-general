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
	
		self.board_size = game.getBoardSize()
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
			num_filters = 32
			self.trunk_1d = nn.Sequential(
				nn.Linear(3*self.num_players*N_COLORS+3*CONSTR_SITE_SIZE, num_filters),
				nn.BatchNorm1d(num_filters),
				nn.ReLU(),
				# nn.Linear(num_filters, num_filters),
				# nn.BatchNorm1d(num_filters),
				# nn.ReLU(),
			)
			self.final_layers_V = nn.Sequential(
				nn.Linear(num_filters, num_filters),
				nn.BatchNorm1d(num_filters),
				nn.ReLU(),
				nn.Linear(num_filters, 1),
			)
			self.final_layers_PI = nn.Sequential(
				nn.Linear(num_filters, self.action_size),
			)
		elif self.version == 8: # Simple version using Embedding
			D, T, S, G = 3, 32, 32, 8
			self.embed = nn.Embedding(num_embeddings=len(CODES_LIST), embedding_dim=D)
			self.conv1d_constr = nn.Conv1d(D, T, kernel_size=3, stride=1, padding=0)
			self.dense_scores = nn.Sequential(
				nn.Linear(N_COLORS*3*self.num_players, S),
				nn.BatchNorm1d(S),
				nn.ReLU(),
				# nn.Linear(S, S),
				# nn.BatchNorm1d(S),
				# nn.ReLU(),
			)
			self.dense_globs = nn.Linear(2, G)
			new_size = CONSTR_SITE_SIZE*(T+S+G) // 8
			self.fused_to_1d = nn.Sequential(
				nn.Flatten(1),
				nn.Linear(CONSTR_SITE_SIZE*(T+S+G), new_size),
				nn.BatchNorm1d(new_size),
				nn.ReLU(),
			)
			self.final_layers_V = nn.Sequential(
				nn.Linear(new_size, new_size),
				nn.BatchNorm1d(new_size),
				nn.ReLU(),
				nn.Linear(new_size, 1),
			)
			self.final_layers_PI = nn.Sequential(
				nn.Linear(new_size, self.action_size),
			)
		elif self.version >= 18: # Less simple version using Embedding
			constants = {
				18:	(3, 32, 32, 8, 16, 32),
				19: (3, 8, 16, 8, 16, 32),
				# INSERT_PARAMS_HERE
			}
			D, T, S, G, B, r = constants[self.version]
			self.embed = nn.Embedding(num_embeddings=len(CODES_LIST), embedding_dim=D)
			self.conv1d_constr = nn.Conv1d(D, T, kernel_size=3, stride=1, padding=0)
			self.dense_scores = nn.Sequential(
				nn.Linear(N_COLORS*3*self.num_players, S),
				# nn.BatchNorm1d(S),
				# nn.ReLU(),
				# nn.Linear(S, S),
				# nn.BatchNorm1d(S),
				# nn.ReLU(),
			)
			self.conv2d_boards = nn.Conv2d(D+2, B, kernel_size=1)
			self.dense_globs = nn.Linear(2, G)
	
			self.final_layers_V = nn.Sequential(
				nn.Flatten(1),
				nn.Linear(CONSTR_SITE_SIZE*(T+S+G), 1),
			)

			self.proj_i = nn.Linear(T+S+G, r)
			self.proj_p = nn.Conv2d(self.num_players*B+G+S, r, kernel_size=1)
			self.U_o = nn.Parameter(torch.randn(6, r))

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
		x = input_data.permute(0, 3, 1, 2)
		# Split data
		split_input_data = x.split([self.num_players, self.num_players, self.num_players, 1 ,1], dim=1)
		boards_descr, boards_height, boards_tileID, per_pl_data, global_data = split_input_data
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
			fused_1d = self.fused_to_1d(fused)
			v = self.final_layers_V(fused_1d)
			pi = torch.where(valid_actions, self.final_layers_PI(fused_1d), self.lowvalue)
		elif self.version >= 18:
			# Work on scores and glob data first
			s1 = F.dropout(self.dense_scores(scores_data.flatten(1)), p=self.args['dropout'], training=self.training) # N,3*NUM_PLAYERS,N_COLORS -> N,S
			g1 = F.dropout(self.dense_globs(globals_data), p=self.args['dropout'], training=self.training) # 2 -> N,G

			# Convert boards_descr to embeddings (need to clamp when input is random)
			boards_descr_long = boards_descr.clamp(min=0., max=len(CODES_LIST)-1).long()
			boards_descr_embed = self.embed(boards_descr_long) # N,2,12,12,D
			boards = torch.cat([boards_descr_embed, boards_height.unsqueeze(-1), boards_tileID.unsqueeze(-1)], dim=-1) # N,2,12,12,D+2
			boards_4d = [boards[:,i,:,:,:].permute(0,3,1,2) for i in range(self.num_players)] # N,D+2,12,12 (num_players times)
			boards_4d = [self.conv2d_boards(boards_4d[i]) for i in range(self.num_players)] # N,B,12,12 (num_players times)

			# Merge boards + scores + global data to a 4D vector
			scores_4d = s1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, CITY_SIZE, CITY_SIZE) # N, S, 12, 12
			glob_4d   = g1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, CITY_SIZE, CITY_SIZE) # N, G, 12, 12
			fused_4d = torch.cat(boards_4d + [scores_4d, glob_4d], dim=1) # N,self.num_players*B+G+S,12,12

			# Convert constrs_site to embeddings (need to clamp when input is random)
			constrs_long = constrs_site.clamp(min=0., max=len(CODES_LIST)-1).long()
			constrs_embed = self.embed(constrs_long) # N,CS,3,D
			constrs_3d = constrs_embed.flatten(start_dim=0, end_dim=1).permute(0, 2, 1) # N*CS, D, 3
			constrs_3d = F.dropout(self.conv1d_constr(constrs_3d), p=self.args['dropout'], training=self.training) # N*CS, T
			constrs_3d = constrs_3d.squeeze(-1).view(constrs_embed.shape[0], constrs_embed.shape[1], -1) # N, CS, T

			# Merge constrs + scores + global data to a 3D vector
			scores_3d = s1.unsqueeze(1).expand(-1, CONSTR_SITE_SIZE, -1) # N, CS, S
			glob_3d = g1.unsqueeze(1).expand(-1, CONSTR_SITE_SIZE, -1) # N, CS, G
			fused_3d = torch.cat([constrs_3d, scores_3d, glob_3d], dim=-1) # N,CS,T+G+S

			# Compute policy
			fused_3d_proj = self.proj_i(fused_3d) # N, CS, r
			fused_3d_proj = fused_3d_proj.unsqueeze(2).unsqueeze(2) # N, CS, 1, 1, r
			fused_4d_proj = self.proj_p(fused_4d) # N, r, 12, 12
			fused_4d_proj = fused_4d_proj.permute(0, 2, 3, 1).unsqueeze(1) # N, 1, 12, 12, r
			
			logits = torch.einsum('nchwt,ot->nchwo', fused_3d_proj*fused_4d_proj, self.U_o) # N, CS, 12, 12, r
			pi = torch.where(valid_actions, logits.flatten(1), self.lowvalue)

			v = self.final_layers_V(fused_3d)
		else:
			raise Exception(f'Unsupported NN version {self.version}')

		return F.log_softmax(pi, dim=1), torch.tanh(v)

