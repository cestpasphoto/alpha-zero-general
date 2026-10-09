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


############################ V85: ENTITY NETWORK ##############################
# The board is read as what it is, sets of cards:
#   - a planet is a SET: scoring only uses the sums of attributes, the characters
#     and the card types, never the slot -> DeepSets sum pooling, exactly
#     invariant to any reordering of the slots
#   - the market is a SET: each card gets its own logits, so reordering the
#     market reorders the policy (exact equivariance, no augmentation needed)
#   - seats are interchangeable except "me" (designation is free, the rules
#     never use seat order) -> one shared encoder per player and a symmetric
#     aggregation of the opponents: permuting the opponents permutes their
#     values and designation logits exactly (what the removed player
#     augmentation of get_symmetries tried, wrongly, to teach)
# Integer fields are decoded inside the net: CARD_TYPE into type and character
# one-hots, the int8 bitfields into bits (who can play, the 80 remaining cards).
# Bits use only Floor on floats, exact and identical in torch and ONNX (no
# integer division, see the ONNX parity guard in GenericNNetWrapper).

def _mlp(sizes, last_zero=False):
	layers = []
	for i in range(len(sizes) - 1):
		layers.append(nn.Linear(sizes[i], sizes[i+1]))
		if i < len(sizes) - 2:
			layers.append(nn.ReLU())
	if last_zero: # uniform policy / zero value at start, gradients still flow
		nn.init.zeros_(layers[-1].weight)
		nn.init.zeros_(layers[-1].bias)
	return nn.Sequential(*layers)

N_ATTR, N_TYPES_EMPTY, N_CHARS = 14, 5, 13   # attributes, types incl. empty, characters
CARD_FEAT = N_ATTR + N_TYPES_EMPTY + N_CHARS  # 32

class TLPEntityNet(nn.Module):
	def __init__(self, num_players, d=32, h=64):
		super().__init__()
		from .TLPLogicNumba import np_all_cards
		n = self.n = num_players
		self.register_buffer('type_ids', torch.arange(N_TYPES_EMPTY, dtype=torch.float32))
		self.register_buffer('char_ids', torch.arange(1, N_CHARS + 1, dtype=torch.float32))
		self.register_buffer('bit_div' , torch.tensor([128., 64., 32., 16., 8., 4., 2., 1.]))
		self.register_buffer('is_me'   , torch.tensor([1.] + [0.] * (n - 1)).view(1, n, 1))
		# features of the 80 cards of the game, grouped by stack: (4, 20, CARD_FEAT)
		self.register_buffer('deck_feat', self._card_features(torch.tensor(np_all_cards, dtype=torch.float32)))

		self.card_enc   = _mlp([CARD_FEAT, h, d])                             # phi, shared by planets and market
		self.player_enc = _mlp([CARD_FEAT + d + 15 + 3, h, d])                # rho of the planet set
		self.stack_enc  = nn.Linear(CARD_FEAT, d)
		self.stack_emb  = nn.Parameter(torch.zeros(4, d))                     # stack identity
		self.global_enc = _mlp([4*d + 4*d + 2, h, d])
		self.value_head = _mlp([2*d, h, 1], last_zero=True)
		self.take_head  = _mlp([3*d, h, 1], last_zero=True)
		self.stack_head = _mlp([2*d, h, 1], last_zero=True)

	def _card_features(self, cards):
		# cards: (..., 15) raw rows -> (..., 32): attributes, type one-hot (0 = empty), character one-hot
		ct = cards[..., 14]
		type_oh = (torch.floor(ct / 25.).unsqueeze(-1) == self.type_ids).float()
		char_oh = ((ct - 100.).unsqueeze(-1) == self.char_ids).float()
		# Attributes clamped to their real range [0, 3] (identity on every real card,
		# face-down included): keeps activations bounded on out-of-distribution
		# boards, where the net has no normalisation layer to do it
		return torch.cat([cards[..., :N_ATTR].clamp(0., 3.), type_oh, char_oh], dim=-1)

	def _bits(self, v):
		# int8 values (as floats) -> (..., 8) bits, MSB first like my_packbits
		u = v - torch.floor(v / 256.) * 256.                     # two's complement -> [0, 255]
		hi = torch.floor(u.unsqueeze(-1) / self.bit_div)
		return hi - 2. * torch.floor(hi / 2.)

	def _masked_mean(self, x, mask):
		return (x * mask).sum(1) / mask.sum(1).clamp(min=1.)

	def forward(self, x, dropout=0.):
		# Every Linear sees a 3D tensor and no reshape names the batch size: the
		# recent torch.onnx exporter otherwise bakes batch=1 into Gemm reshapes
		n = self.n
		meta, market_raw, scores = x[:, 0, :], x[:, 1:n+1, :], x[:, n+1:2*n+1, :]
		# Real scores lie in [-12, 127] (volcano penalty >= -12 in column 0, other
		# columns >= 0, int8 above): identity in play, bounded on synthetic boards
		scores = scores.clamp(-16., 127.)

		# meta row: round, who can play, remaining cards
		round_ = meta[:, 0:1] / (16. * n)                                             # (B, 1)
		can_play = self._bits(meta[:, 2])[:, :n].unsqueeze(-1)                        # (B, n, 1)
		avail = self._bits(meta[:, 3:13]).reshape(-1, 4, 1, 20)                       # (B, 4, 1, 20)
		pool = torch.matmul(avail, self.deck_feat).squeeze(2) / 8.                    # (B, 4, 32)

		# planets: DeepSets over the 16 slots
		cf = self._card_features(x[:, 2*n+1:18*n+1, :])                               # (B, 16n, 32)
		occupied = (cf[..., N_ATTR:N_ATTR+1] < 0.5).float()                           # type != empty
		emb = (self.card_enc(cf) * occupied).reshape(-1, n, 16, self.card_enc[-1].out_features).sum(2)
		raw_sum = cf.reshape(-1, n, 16, CARD_FEAT).sum(2)
		total = scores.sum(-1, keepdim=True) / 32.
		is_me = self.is_me + 0. * can_play                                            # (B, n, 1), broadcast
		players = self.player_enc(torch.cat([raw_sum / 4., emb / 4., scores / 16., total, can_play, is_me], dim=-1))  # (B, n, d)

		# market
		mf = self._card_features(market_raw)                                          # (B, n, 32)
		m_mask = (mf[..., N_ATTR:N_ATTR+1] < 0.5).float()
		market = self.card_enc(mf) * m_mask                                           # (B, n, d)
		choose_phase = 1. - m_mask.amax(1)                                             # (B, 1)

		# stacks to choose from
		stacks = self.stack_enc(pool) + self.stack_emb                                # (B, 4, d)

		# global context: me, opponents (symmetric), market, stacks
		opp = players[:, 1:, :]
		g = self.global_enc(torch.cat([players[:, 0, :], opp.mean(1), opp.amax(1),
		                               self._masked_mean(market, m_mask), stacks.mean(1), stacks.amax(1),
		                               market.amax(1), players.amax(1), round_, choose_phase], dim=-1))
		g = F.dropout(F.relu(g), p=dropout, training=self.training)                  # (B, d)
		g1 = g.unsqueeze(1)

		# value: one shared head per seat
		v = self.value_head(torch.cat([players, g1.expand(-1, n, -1)], dim=-1)).squeeze(-1)

		# policy, take card i and designate seat d: action i*n + d (row-major over (i, d))
		pair = torch.cat([market.unsqueeze(2).expand(-1, n, n, -1),
		                  players.unsqueeze(1).expand(-1, n, n, -1),
		                  g.unsqueeze(1).unsqueeze(1).expand(-1, n, n, -1)], dim=-1)
		take = self.take_head(pair.reshape(-1, n * n, pair.shape[-1])).squeeze(-1)  # (B, n*n)
		# policy, choose stack t: action n*n + t
		choose = self.stack_head(torch.cat([stacks, g1.expand(-1, 4, -1)], dim=-1)).squeeze(-1)
		return torch.cat([take, choose], dim=1), v


class TLPNNet(nn.Module):
	def __init__(self, game, args):
		# game params
		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.args = args
		self.version = args['nn_version']
		super(TLPNNet, self).__init__()
		if self.version == 85: # Entity network (DeepSets planets, per-card policy, shared per-seat value)
			self.entity = TLPEntityNet(self.num_players)

		elif self.version == 80: # Very small version using MobileNetV3 building blocks
			self.first_layer = LinearNormActivation(self.nb_vect, self.nb_vect, None)
			confs  = []
			confs += [InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 15, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			head_PI = [
				InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 15, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(self.nb_vect*15, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 15, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(self.nb_vect*15, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

		elif self.version == 81: # Not that small variant of V80
			self.first_layer = LinearNormActivation(self.nb_vect, self.nb_vect, None)
			confs  = []
			confs += [
				InvertedResidual1d(self.nb_vect, 4*self.nb_vect, self.nb_vect, 15, False, "RE"),
				InvertedResidual1d(self.nb_vect, 4*self.nb_vect, self.nb_vect, 15, False, "RE"),
			]
			self.trunk = nn.Sequential(*confs)

			head_PI = [
				InvertedResidual1d(self.nb_vect, 4*self.nb_vect, self.nb_vect, 15, True, "HS", setype='avg'),
				InvertedResidual1d(self.nb_vect, 4*self.nb_vect, self.nb_vect, 15, True, "HS", setype='avg'),
				nn.Flatten(1),
				nn.Linear(self.nb_vect*15, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(self.nb_vect, 4*self.nb_vect, self.nb_vect, 15, True, "HS", setype='avg'),
				InvertedResidual1d(self.nb_vect, 4*self.nb_vect, self.nb_vect, 15, True, "HS", setype='avg'),
				nn.Flatten(1),
				nn.Linear(self.nb_vect*15, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

		elif self.version == 82: # Even smaller than 80
			self.first_layer = LinearNormActivation(self.nb_vect, self.nb_vect, None)
			confs  = []
			confs += [InvertedResidual1d(self.nb_vect, 2*self.nb_vect, self.nb_vect, 15, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			head_PI = [
				InvertedResidual1d(self.nb_vect, 2*self.nb_vect, self.nb_vect, 15, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(self.nb_vect*15, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(self.nb_vect, 2*self.nb_vect, self.nb_vect, 15, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(self.nb_vect*15, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

		elif self.version == 83: # Even even smaller than 80
			self.first_layer = LinearNormActivation(self.nb_vect, self.nb_vect, None)
			confs  = []
			confs += [InvertedResidual1d(self.nb_vect, int(1.5*self.nb_vect), self.nb_vect, 15, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			head_PI = [
				InvertedResidual1d(self.nb_vect, int(1.5*self.nb_vect), self.nb_vect, 15, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(self.nb_vect*15, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(self.nb_vect, int(1.5*self.nb_vect), self.nb_vect, 15, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(self.nb_vect*15, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

		self.register_buffer('lowvalue', torch.FloatTensor([-1e8]))
		# Former _init() removed: it walked self.__dict__, where nn.Module does not
		# store submodules, so it never ran (and would have crashed on the bias-free
		# Linear layers). PyTorch default init was, and stays, the effective one.

	def forward(self, input_data, valid_actions):
		# input_data is (N, H, C) typically (N, 55, 15)
		if self.version == 85:
			x = input_data.view(-1, self.nb_vect, self.vect_dim)
			pi, v = self.entity(x, dropout=self.args['dropout'])
			pi = torch.where(valid_actions, pi, self.lowvalue)

		elif self.version in [80, 81, 82, 83]:
			x = input_data.view(-1, self.nb_vect, self.vect_dim) # no transpose
			x = self.first_layer(x)
			x = F.dropout(self.trunk(x), p=self.args['dropout'], training=self.training)
			v = self.output_layers_V(x)
			pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)

		else:
			raise Exception(f'Unsupported NN version {self.version}')

		return F.log_softmax(pi, dim=1), torch.tanh(v)
