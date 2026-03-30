import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _make_divisible

class EntityEmbedding(nn.Module):
	"""Embeds groups of rows from the 56x7 board into tokens of uniform dimension.
	Each semantic group gets its own Linear(7, d) encoder.
	Cards within the same tier share weights (weight sharing for symmetry).
	"""
	def __init__(self, d_model, num_players):
		super().__init__()
		n = num_players
		self.num_players = n
		self.d_model = d_model

		# Group definitions: (start_row, end_row_exclusive, stride)
		# stride=2 means we take pairs of rows and flatten them to 14 inputs
		# stride=1 means single rows of 7 inputs

		# Shared encoders by type
		self.enc_bank       = nn.Linear(7, d_model)           # 1 token: bank
		self.enc_card       = nn.Linear(14, d_model)          # 12 tokens: visible cards (cost+gain = 2 rows → 14)
		self.enc_noble      = nn.Linear(7, d_model)           # n+1 tokens: nobles in bank
		self.enc_player_gem = nn.Linear(7, d_model)           # n tokens: player gems
		self.enc_player_crd = nn.Linear(7, d_model)           # n tokens: player card counts
		self.enc_reserve    = nn.Linear(14, d_model)          # 3*n tokens: reserved cards (cost+gain)

		# Type embeddings (learned)
		self.type_emb = nn.Embedding(6, d_model)  # 6 types: bank, card, noble, pgem, pcrd, reserve

		# Precompute sizes
		self.n_nobles = n + 1
		self.n_reserve = 3 * n

		# Total tokens = 1 + 12 + (n+1) + n + n + 3*n = 14 + 6*n
		self.num_tokens = 1 + 12 + (n+1) + n + n + 3*n

	def forward(self, x):
		# x: (B, 56, 7) — but actual size is (B, nb_vect, 7)
		B = x.shape[0]
		n = self.num_players
		tokens = []
		types = []

		# 1. Bank: row 0
		tokens.append(self.enc_bank(x[:, 0:1, :]))                  # (B,1,d)
		types.append(torch.zeros(B, 1, dtype=torch.long, device=x.device))

		# 2. Visible cards: rows 1-24, pairs of rows (cost, gain)
		cards = x[:, 1:25, :].reshape(B, 12, 14)                    # (B,12,14)
		tokens.append(self.enc_card(cards))                          # (B,12,d)
		types.append(torch.ones(B, 12, dtype=torch.long, device=x.device))

		# 3. Nobles: rows 31 to 31+n+1
		noble_start = 31
		nobles = x[:, noble_start:noble_start+self.n_nobles, :]     # (B,n+1,7)
		tokens.append(self.enc_noble(nobles))                        # (B,n+1,d)
		types.append(torch.full((B, self.n_nobles), 2, dtype=torch.long, device=x.device))

		# 4. Player gems: rows 32+n to 32+2n
		pg_start = 32 + n
		pgems = x[:, pg_start:pg_start+n, :]                        # (B,n,7)
		tokens.append(self.enc_player_gem(pgems))                    # (B,n,d)
		types.append(torch.full((B, n), 3, dtype=torch.long, device=x.device))

		# 5. Player cards: rows 32+3n+n² to 32+4n+n²
		pc_start = 32 + 3*n + n*n
		pcards = x[:, pc_start:pc_start+n, :]                       # (B,n,7)
		tokens.append(self.enc_player_crd(pcards))                   # (B,n,d)
		types.append(torch.full((B, n), 4, dtype=torch.long, device=x.device))

		# 6. Reserved cards: rows 32+4n+n² to 32+10n+n², pairs of rows
		rs_start = 32 + 4*n + n*n
		reserves = x[:, rs_start:rs_start+6*n, :].reshape(B, 3*n, 14)  # (B,3n,14)
		tokens.append(self.enc_reserve(reserves))                       # (B,3n,d)
		types.append(torch.full((B, self.n_reserve), 5, dtype=torch.long, device=x.device))

		# Concatenate all tokens
		all_tokens = torch.cat(tokens, dim=1)                       # (B, num_tokens, d)
		all_types  = torch.cat(types, dim=1)                         # (B, num_tokens)

		# Add type embeddings
		all_tokens = all_tokens + self.type_emb(all_types)

		return all_tokens

class StructuredPolicyHead(nn.Module):
	"""
	Bifurcated policy head:
	- Card actions (buy visible 0-11, reserve visible 12-23, reserve deck 24-26,
	  buy reserve 27-29): one logit per relevant token via Linear(d→1)
	- Gem actions (30-79) + pass (80): MLP from pooled representation
	"""
	def __init__(self, d_model, num_tokens, n_card_tokens=12, n_reserve_tokens=9):
		super().__init__()
		self.n_card_tokens = n_card_tokens
		self.n_reserve_tokens = n_reserve_tokens

		# Per-token logit for card actions
		self.buy_visible_head  = nn.Linear(d_model, 1)     # 12 logits for buy visible
		self.rsv_visible_head  = nn.Linear(d_model, 1)     # 12 logits for reserve visible
		self.buy_reserve_head  = nn.Linear(d_model, 1)     # 3*n logits for buy from reserve

		# Reserve from deck: 3 logits from global representation
		# Gem actions: 51 logits (30 different + 5 identical + 15 give + 5 give identical + 1 pass = 51)
		# Total non-card actions: 3 + 50 + 1 = 54... actually let's keep it simple:
		# 3 (rsv deck) + 30 (get gems) + 20 (give gems) + 1 (pass) = 54
		self.gem_head = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.ReLU(),
			nn.Linear(d_model, 3 + 50 + 1),  # rsv_deck(3) + get_gems(30) + give_gems(20) + pass(1)
		)

	def forward(self, tokens, num_players):
		"""
		tokens: (B, T, d) — output of trunk
		Returns: (B, 81) logits
		"""
		B, T, d = tokens.shape

		# Card tokens are at indices 1..12
		card_tokens = tokens[:, 1:1+self.n_card_tokens, :]            # (B,12,d)
		buy_visible_logits = self.buy_visible_head(card_tokens).squeeze(-1)  # (B,12)
		rsv_visible_logits = self.rsv_visible_head(card_tokens).squeeze(-1)  # (B,12)

		# Reserve tokens are the last n_reserve_tokens, but only first 3 are for player 0
		res_start = T - self.n_reserve_tokens
		# Player 0's reserves are the first 3 of the reserve tokens
		p0_reserve_tokens = tokens[:, res_start:res_start+3, :]       # (B,3,d)
		buy_reserve_logits = self.buy_reserve_head(p0_reserve_tokens).squeeze(-1)  # (B,3)

		# Global pooling for gem/deck actions
		global_pool = tokens.mean(dim=1)                               # (B,d)
		other_logits = self.gem_head(global_pool)                      # (B,54)
		# Split: rsv_deck(3), get_gems(30), give_gems(20), pass(1)
		rsv_deck, get_gems, give_gems, do_pass = other_logits.split([3, 30, 20, 1], dim=1)

		# Assemble in action order:
		# 0-11: buy visible, 12-23: rsv visible, 24-26: rsv deck,
		# 27-29: buy reserve, 30-59: get gems, 60-79: give gems, 80: pass
		pi = torch.cat([
			buy_visible_logits,     # 12
			rsv_visible_logits,     # 12
			rsv_deck,               # 3
			buy_reserve_logits,     # 3
			get_gems,               # 30
			give_gems,              # 20
			do_pass,                # 1
		], dim=1)                   # (B, 81)

		return pi

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



class SplendorNNet(nn.Module):
	def __init__(self, game, args):
		# game params
		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.args = args
		self.version = args['nn_version']

		super(SplendorNNet, self).__init__()
		if self.version == 80: # Very small version using MobileNetV3 building blocks
			self.first_layer = LinearNormActivation(self.nb_vect, self.nb_vect, None)
			confs  = []
			confs += [InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 7, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			head_PI = [
				InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 7, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(self.nb_vect*7, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 7, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(self.nb_vect*7, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

		elif self.version == 92:  # Micro-Transformer (Fixed)
			d = 56
			self.embedding = EntityEmbedding(d, self.num_players)
			T = self.embedding.num_tokens
			self.pos_emb = nn.Parameter(torch.randn(1, T, d) * 0.02)
			encoder_layer = nn.TransformerEncoderLayer(
				d_model=d,
				nhead=4,              # make sure that d is divisible by nhead
				dim_feedforward=d*3,
				dropout=self.args.get('dropout', 0.1),
				activation='gelu',
				batch_first=True,
				norm_first=True
			)
			self.trunk = nn.TransformerEncoder(encoder_layer, num_layers=3)

			self.head_pi = StructuredPolicyHead(d, T, n_card_tokens=12, n_reserve_tokens=3*self.num_players)
			self.head_v = nn.Sequential(
				nn.LayerNorm(d),
				nn.Linear(d, self.num_players),
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
		# input_data is (N, H, C) typically (N, 56, 7)
		x = input_data.view(-1, self.nb_vect, self.vect_dim) # no transpose

		if self.version in [80]:
			x = self.first_layer(x)
			x = F.dropout(self.trunk(x), p=self.args['dropout'], training=self.training)
			v = self.output_layers_V(x)
			pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)

		elif self.version == 92:
			tokens = self.embedding(x)
			tokens = tokens + self.pos_emb
			tokens = self.trunk(tokens)
			v = self.head_v(tokens.mean(dim=1))
			pi = torch.where(valid_actions, self.head_pi(tokens, self.num_players), self.lowvalue)

		else:
			raise Exception(f'Unsupported NN version {self.version}')

		return F.log_softmax(pi, dim=1), torch.tanh(v)

