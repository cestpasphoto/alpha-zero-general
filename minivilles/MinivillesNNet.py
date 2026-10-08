import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _make_divisible

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


class MinivillesNNet(nn.Module):
	def __init__(self, game, args):
		# game params
		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.args = args
		self.version = args['nn_version']

		self.scdiff_size = 2 * (52-0) + 1
		self.num_scdiffs = self.num_players

		super(MinivillesNNet, self).__init__()

		if self.version == 82:
			self.first_layer = LinearNormActivation(self.nb_vect, self.nb_vect, None)
			confs  = []
			confs += [InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 2, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			head_PI = [
				InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 2, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(self.nb_vect*2, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 2, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(self.nb_vect*2, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

		elif self.version == 83:
			# V83: Temporal MLP - Flattens the entire board (58 features * 2 history states = 116)
			self.flat_size = self.nb_vect * self.vect_dim
			
			self.trunk = nn.Sequential(
				nn.Linear(self.flat_size, 256),
				nn.LayerNorm(256),
				nn.SiLU(),
				nn.Linear(256, 256),
				nn.LayerNorm(256),
				nn.SiLU(),
				nn.Linear(256, 128),
				nn.LayerNorm(128),
				nn.SiLU()
			)
			
			self.output_layers_PI = nn.Sequential(
				nn.Linear(128, 64),
				nn.SiLU(),
				nn.Linear(64, self.action_size)
			)
			
			self.output_layers_V = nn.Sequential(
				nn.Linear(128, 64),
				nn.SiLU(),
				nn.Linear(64, self.num_players)
			)

		elif self.version == 84:
			# V84: per-player tokens (shared encoder) + global token, small self-attention,
			# value read per player token (seat-equivariant), policy from global + current player.
			n, d = self.num_players, 64
			self.p_feat = 2 * 20 + 4  # (money, 15 cards, 4 monuments) x 2 slots + 4 combo products
			scale = torch.ones(self.nb_vect)
			scale[0], scale[1], scale[2] = 1/64, 1/12, 1/3   # round, last dice, player_state
			scale[3:18] = 1/6                                 # market
			scale[18:18+n] = 1/32                             # money
			scale[18+n:18+16*n] = 1/4                         # cards
			self.register_buffer('row_scale', scale.view(1, -1, 1))
			self.glob_enc   = nn.Sequential(nn.Linear(18 * 2, d), nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d))
			self.player_enc = nn.Sequential(nn.Linear(self.p_feat, d), nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d))
			self.seat_emb   = nn.Parameter(0.02 * torch.randn(n, d))
			layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2*d, dropout=0.0, batch_first=True, norm_first=True)
			self.mixer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
			self.out_norm = nn.LayerNorm(d)
			self.output_layers_PI = nn.Sequential(nn.Linear(2*d, d), nn.SiLU(), nn.Linear(d, self.action_size))
			self.output_layers_V  = nn.Sequential(nn.Linear(d, d//2), nn.SiLU(), nn.Linear(d//2, 1))

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
		# input_data is (N, H, C) typically (N, 58, 2)
		x = input_data.view(-1, self.nb_vect, self.vect_dim) # no transpose
		if self.version in [82]:
			x = self.first_layer(x)
			x = F.dropout(self.trunk(x), p=self.args['dropout'], training=self.training)
			v = self.output_layers_V(x)
			pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)
			
		elif self.version == 83:
			x = F.dropout(self.trunk(x.flatten(1)), p=self.args.get('dropout', 0.1), training=self.training) # was missing flatten
			v = self.output_layers_V(x)
			pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)

		elif self.version == 84:
			v, pi_logits = self._forward_v84(x)
			pi = torch.where(valid_actions, pi_logits, self.lowvalue)

		else:
			raise Exception(f'Unsupported NN version {self.version}')

		return F.log_softmax(pi, dim=1), torch.tanh(v)

	def _forward_v84(self, x):
		# x: (N, R, 2) raw counts, canonical form (current player is seat 0)
		n = self.num_players
		x = x * self.row_scale
		glob  = x[:, :18, :].flatten(1)                                   # (N, 36)
		money = x[:, 18:18+n, :]                                          # (N, n, 2)
		cards = x[:, 18+n:18+16*n, :].reshape(-1, n, 15, 2)               # (N, n, 15, 2)
		monum = x[:, 18+16*n:18+20*n, :].reshape(-1, n, 4, 2)             # (N, n, 4, 2)
		c = cards[..., 0]
		# Multiplicative income terms an MLP learns poorly (indexes from MinivillesLogicNumba)
		combos = torch.stack([
			c[..., 9]  * c[..., 1],                                       # cheese factory x ranch
			c[..., 10] * (c[..., 5] + c[..., 11]),                        # furniture x (forest + mine)
			c[..., 14] * (c[..., 0] + c[..., 13]),                        # market x (wheat + orchard)
			monum[..., 1, 0] * (c[..., 2] + c[..., 3] + c[..., 4] + c[..., 12]),  # mall x (cup + bread)
		], dim=-1)                                                        # (N, n, 4)
		p = torch.cat([money, cards.flatten(2), monum.flatten(2), combos], dim=-1)  # (N, n, 44)
		tokens = torch.cat([self.glob_enc(glob).unsqueeze(1), self.player_enc(p) + self.seat_emb], dim=1)
		h = self.out_norm(self.mixer(tokens))                             # (N, n+1, d)
		h = F.dropout(h, p=self.args.get('dropout', 0.), training=self.training)
		v = self.output_layers_V(h[:, 1:, :]).squeeze(-1)                 # (N, n)
		pi = self.output_layers_PI(torch.cat([h[:, 0, :], h[:, 1, :]], dim=-1))
		return v, pi
