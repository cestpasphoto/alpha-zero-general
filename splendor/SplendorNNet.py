import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _make_divisible

# =========================================================================
# 1. ENTITY PROCESSING MODULES
# =========================================================================

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

		# Shared encoders by type
		self.enc_bank       = nn.Linear(7, d_model)           # 1 token: bank
		self.enc_deck       = nn.Linear(7, d_model)           # 3 tokens: tier 1, 2, 3 counts
		self.enc_card       = nn.Linear(14, d_model)          # 12 tokens: visible cards (cost+gain)
		self.enc_noble      = nn.Linear(7, d_model)           # n+1 tokens: nobles in bank
		self.enc_player_gem = nn.Linear(7, d_model)           # n tokens: player gems
		self.enc_player_nbl = nn.Linear(7, d_model)           # n tokens: player nobles earned (summed)
		self.enc_player_crd = nn.Linear(7, d_model)           # n tokens: player card counts
		self.enc_reserve    = nn.Linear(14, d_model)          # 3*n tokens: reserved cards (cost+gain)

		# Type embeddings (learned)
		self.type_emb = nn.Embedding(8, d_model)  # 8 semantic types

		self.n_nobles = n + 1
		self.n_reserve = 3 * n

		# Total tokens = 1 (bank) + 3 (decks) + 12 (cards) + (n+1) (nobles) + n (pgems) + n (pnobles) + n (pcards) + 3*n (reserves)
		self.num_tokens = 16 + self.n_nobles + 6 * n

	def forward(self, x):
		# x: (B, nb_vect, 7)
		B = x.shape[0]
		n = self.num_players
		tokens = []
		types = []

		# 1. Bank: row 0
		tokens.append(self.enc_bank(x[:, 0:1, :]))                  # (B,1,d)
		types.append(torch.full((B, 1), 0, dtype=torch.long, device=x.device))

		# 2. Decks counts: rows 25, 27, 29 (skipping bitfields)
		tokens.append(self.enc_deck(x[:, 25:31:2, :]))              # (B,3,d)
		types.append(torch.full((B, 3), 1, dtype=torch.long, device=x.device))

		# 3. Visible cards: rows 1-24, pairs of rows (cost, gain)
		cards = x[:, 1:25, :].reshape(B, 12, 14)                    # (B,12,14)
		tokens.append(self.enc_card(cards))                          # (B,12,d)
		types.append(torch.full((B, 12), 2, dtype=torch.long, device=x.device))

		# 4. Nobles: rows 31 to 31+n+1
		noble_start = 31
		nobles = x[:, noble_start:noble_start+self.n_nobles, :]     # (B,n+1,7)
		tokens.append(self.enc_noble(nobles))                        # (B,n+1,d)
		types.append(torch.full((B, self.n_nobles), 3, dtype=torch.long, device=x.device))

		# 5. Player gems: rows 32+n to 32+2n
		pg_start = 32 + n
		pgems = x[:, pg_start:pg_start+n, :]                        # (B,n,7)
		tokens.append(self.enc_player_gem(pgems))                    # (B,n,d)
		types.append(torch.full((B, n), 4, dtype=torch.long, device=x.device))

		# 6. Player nobles earned: sums all nobles for each player
		pn_start = 32 + 2*n
		pnobles = x[:, pn_start:pn_start+n*self.n_nobles, :].reshape(B, n, self.n_nobles, 7).sum(dim=2) # (B,n,7)
		tokens.append(self.enc_player_nbl(pnobles))                  # (B,n,d)
		types.append(torch.full((B, n), 5, dtype=torch.long, device=x.device))

		# 7. Player cards count
		pc_start = 32 + 3*n + n*n
		pcards = x[:, pc_start:pc_start+n, :]                       # (B,n,7)
		tokens.append(self.enc_player_crd(pcards))                   # (B,n,d)
		types.append(torch.full((B, n), 6, dtype=torch.long, device=x.device))

		# 8. Reserved cards
		rs_start = 32 + 4*n + n*n
		reserves = x[:, rs_start:rs_start+6*n, :].reshape(B, 3*n, 14)  # (B,3n,14)
		tokens.append(self.enc_reserve(reserves))                       # (B,3n,d)
		types.append(torch.full((B, self.n_reserve), 7, dtype=torch.long, device=x.device))

		# Concatenate all tokens
		all_tokens = torch.cat(tokens, dim=1)                       # (B, num_tokens, d)
		all_types  = torch.cat(types, dim=1)                         # (B, num_tokens)

		# Add type embeddings
		all_tokens = all_tokens + self.type_emb(all_types)

		return all_tokens


class StructuredPolicyHead(nn.Module):
	def __init__(self, d_model, num_tokens, n_card_tokens=12, n_reserve_tokens=9):
		super().__init__()
		self.n_card_tokens = n_card_tokens
		self.n_reserve_tokens = n_reserve_tokens

		self.buy_visible_head  = nn.Linear(d_model, 1)     
		self.rsv_visible_head  = nn.Linear(d_model, 1)     
		self.buy_reserve_head  = nn.Linear(d_model, 1)     

		# FIX: Flatten all tokens instead of pooling them to preserve exact arithmetic
		self.gem_head = nn.Sequential(
			nn.Flatten(1),
			nn.Linear(num_tokens * d_model, d_model),
			nn.ReLU(),
			nn.Linear(d_model, 3 + 30 + 20 + 1),  # rsv_deck(3) + get_gems(30) + give_gems(20) + pass(1)
		)

	def forward(self, tokens, num_players):
		B, T, d = tokens.shape

		card_tokens = tokens[:, 4:4+self.n_card_tokens, :]
		buy_visible_logits = self.buy_visible_head(card_tokens).squeeze(-1)
		rsv_visible_logits = self.rsv_visible_head(card_tokens).squeeze(-1)

		res_start = T - self.n_reserve_tokens
		p0_reserve_tokens = tokens[:, res_start:res_start+3, :]
		buy_reserve_logits = self.buy_reserve_head(p0_reserve_tokens).squeeze(-1)

		# FIX: Pass the raw tokens to let the gem_head flatten and read them precisely
		other_logits = self.gem_head(tokens)
		rsv_deck, get_gems, give_gems, do_pass = other_logits.split([3, 30, 20, 1], dim=1)

		pi = torch.cat([
			buy_visible_logits,
			rsv_visible_logits,
			rsv_deck,
			buy_reserve_logits,
			get_gems,
			give_gems,
			do_pass,
		], dim=1)

		return pi

# =========================================================================
# 2. BUILDING BLOCKS (V80, V90, V91)
# =========================================================================

class LinearNormActivation(nn.Module):
	# Ajout de use_bn=True
	def __init__(self, in_size, out_size, activation_layer, depthwise=False, channels=None, use_bn=True):
		super().__init__()
		self.linear     = nn.Linear(in_size, out_size, bias=False)
		
		# Condition sur le BatchNorm
		if use_bn:
			self.norm = nn.BatchNorm1d(channels if depthwise else out_size)
		else:
			self.norm = nn.Identity()
			
		self.activation = activation_layer(inplace=True) if activation_layer is not None else nn.Identity()
		self.depthwise = depthwise

	def forward(self, input):
		if self.depthwise:
			result = self.linear(input)
		else:
			result = self.linear(input.transpose(-1, -2)).transpose(-1, -2)
		result = self.norm(result)
		return self.activation(result)


class SqueezeExcitation1d(nn.Module):
	def __init__(self, input_channels, squeeze_channels, scale_activation, setype='avg'):
		super().__init__()
		self.avgpool = torch.nn.AdaptiveAvgPool1d(1) if setype == 'avg' else torch.nn.AdaptiveMaxPool1d(1)
		self.fc1 = nn.Linear(input_channels, squeeze_channels)
		self.activation = nn.ReLU()
		self.fc2 = torch.nn.Linear(squeeze_channels, input_channels)
		self.scale_activation = scale_activation()

	def forward(self, input):
		scale = self.avgpool(input)
		scale = self.fc1(scale.transpose(-1, -2)).transpose(-1, -2)
		scale = self.activation(scale)
		scale = self.fc2(scale.transpose(-1, -2)).transpose(-1, -2)
		return self.scale_activation(scale) * input

class InvertedResidual1d(nn.Module):
	# Ajout de use_bn=True
	def __init__(self, in_channels, exp_channels, out_channels, kernel, use_hs, use_se, setype='avg', use_bn=True):
		super().__init__()
		self.use_res_connect = (in_channels == out_channels)
		activation_layer = nn.Hardswish if use_hs else nn.ReLU

		# On passe le flag use_bn aux sous-couches
		self.expand = LinearNormActivation(in_channels, exp_channels, activation_layer, use_bn=use_bn) if exp_channels != in_channels else nn.Identity()
		self.depthwise = LinearNormActivation(kernel, kernel, activation_layer, depthwise=True, channels=exp_channels, use_bn=use_bn)
		self.se = SqueezeExcitation1d(exp_channels, _make_divisible(exp_channels // 4, 8), nn.Hardsigmoid, setype) if use_se else nn.Identity()
		self.project = LinearNormActivation(exp_channels, out_channels, activation_layer=None, use_bn=use_bn)

	def forward(self, input):
		res = self.project(self.se(self.depthwise(self.expand(input))))
		return res + input if self.use_res_connect else res

class MLPMixerBlock(nn.Module):
	def __init__(self, num_tokens, d_model, dropout=0.1):
		super().__init__()
		self.norm1 = nn.LayerNorm(d_model)
		self.token_mixing = nn.Sequential(
			nn.Linear(num_tokens, num_tokens),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(num_tokens, num_tokens)
		)
		self.norm2 = nn.LayerNorm(d_model)
		self.channel_mixing = nn.Sequential(
			nn.Linear(d_model, d_model * 4),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(d_model * 4, d_model)
		)
	
	def forward(self, x):
		res = x
		x = self.norm1(x).transpose(1, 2)
		x = self.token_mixing(x).transpose(1, 2)
		x = x + res
		
		res = x
		x = self.norm2(x)
		x = self.channel_mixing(x)
		return x + res

class BilinearAugmentation(nn.Module):
	def __init__(self, d_model):
		super().__init__()
		self.proj = nn.Linear(d_model, d_model)
		
	def forward(self, tokens, player_idx):
		# Extracts Player 0's gems and projects the dot interaction over all tokens
		p_gems = tokens[:, player_idx:player_idx+1, :] # (B, 1, d)
		interaction = p_gems * tokens # (B, T, d)
		return tokens + self.proj(interaction)

# =========================================================================
# 3. MAIN NETWORK ARCHITECTURE
# =========================================================================

class SplendorNNet(nn.Module):
	def __init__(self, game, args):
		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.args = args
		self.version = args['nn_version']
		super(SplendorNNet, self).__init__()

		if self.version == 80: # Very small version using MobileNetV3 building blocks
			self.first_layer = LinearNormActivation(self.nb_vect, self.nb_vect, None)
			confs = [InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 7, False, "RE")]
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
		
		elif self.version == 81: # V81: Channel-Isolated (Orthogonalité des couleurs)
			# We isolate the 7 columns (colors/points) before mixing them
			self.first_layer = LinearNormActivation(self.nb_vect, self.nb_vect, None)
			
			# Independent processing per color channel using grouped 1D convolutions
			self.color_processing = nn.Conv1d(in_channels=7, out_channels=28, kernel_size=1, groups=7)
			
			confs = [InvertedResidual1d(self.nb_vect, 2*self.nb_vect, self.nb_vect, 28, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			self.output_layers_PI = nn.Sequential(
				nn.Flatten(1),
				nn.Linear(self.nb_vect * 28, self.action_size)
			)
			self.output_layers_V = nn.Sequential(
				nn.Flatten(1),
				nn.Linear(self.nb_vect * 28, self.num_players)
			)
		
		elif self.version == 90:  # Entity-Mixer
			d = 56
			self.embedding = EntityEmbedding(d, self.num_players)
			T = self.embedding.num_tokens
			
			self.trunk = nn.Sequential(*[MLPMixerBlock(T, d, self.args.get('dropout', 0.1)) for _ in range(3)])
			self.head_pi = StructuredPolicyHead(d, T, n_card_tokens=12, n_reserve_tokens=3*self.num_players)
			self.head_v = nn.Sequential(
				nn.Flatten(1),
				nn.Linear(T * d, d),
				nn.ReLU(),
				nn.Linear(d, self.num_players)
			)

		elif self.version == 91:  # Entity-Mixer + Bilinear
			d = 56
			self.embedding = EntityEmbedding(d, self.num_players)
			T = self.embedding.num_tokens
			
			# Index of Player 0 gems token = 1 (bank) + 3 (decks) + 12 (cards) + (n+1) (nobles)
			self.player0_gem_idx = 17 + self.num_players
			self.bilinear_aug = BilinearAugmentation(d)
			
			self.trunk = nn.Sequential(*[MLPMixerBlock(T, d, self.args.get('dropout', 0.1)) for _ in range(3)])
			self.head_pi = StructuredPolicyHead(d, T, n_card_tokens=12, n_reserve_tokens=3*self.num_players)
			self.head_v = nn.Sequential(
				nn.Flatten(1),
				nn.Linear(T * d, d),
				nn.ReLU(),
				nn.Linear(d, self.num_players)
			)

		elif self.version == 92:  # Micro-Transformer (Fixed)
			d = 56
			self.embedding = EntityEmbedding(d, self.num_players)
			T = self.embedding.num_tokens
			self.pos_emb = nn.Parameter(torch.randn(1, T, d) * 0.02)
			encoder_layer = nn.TransformerEncoderLayer(
				d_model=d,
				nhead=4,
				dim_feedforward=d*3,
				dropout=self.args.get('dropout', 0.1),
				activation='gelu',
				batch_first=True,
				norm_first=True
			)
			self.trunk = nn.TransformerEncoder(encoder_layer, num_layers=3)
			self.head_pi = StructuredPolicyHead(d, T, n_card_tokens=12, n_reserve_tokens=3*self.num_players)
			self.head_v = nn.Sequential(
				nn.Flatten(1),
				nn.Linear(T * d, d),
				nn.ReLU(),
				nn.Linear(d, self.num_players)
			)

		elif self.version == 100: # V100 (V80 Large): Deeper trunk sans BatchNorm
			# On peut aussi désactiver le BN sur la toute première couche
			self.first_layer = LinearNormActivation(self.nb_vect, self.nb_vect, None, use_bn=False)
			
			confs = [
				InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 7, False, "RE", use_bn=False),
				InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 7, False, "RE", use_bn=False),
				InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 7, False, "RE", use_bn=False)
			]
			self.trunk = nn.Sequential(*confs)

			head_PI = [
				InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 7, True, "HS", setype='max', use_bn=False),
				nn.Flatten(1),
				nn.Linear(self.nb_vect*7, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(self.nb_vect, 3*self.nb_vect, self.nb_vect, 7, True, "HS", setype='max', use_bn=False),
				nn.Flatten(1),
				nn.Linear(self.nb_vect*7, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

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
		x = input_data.view(-1, self.nb_vect, self.vect_dim) # no transpose

		if self.version in [80, 100]:
			x = self.first_layer(x)
			x = F.dropout(self.trunk(x), p=self.args['dropout'], training=self.training)
			v = self.output_layers_V(x)
			pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)

		elif self.version == 81: # Channel-Isolated
			x = self.first_layer(x)
			# Transpose to (Batch, Channels, Length) for standard Conv1d
			x = x.transpose(1, 2)
			x = F.relu(self.color_processing(x))
			x = x.transpose(1, 2) # Back to (Batch, Length, Channels)
			
			x = F.dropout(self.trunk(x), p=self.args['dropout'], training=self.training)
			v = self.output_layers_V(x)
			pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)

		elif self.version == 90:
			tokens = self.embedding(x)
			tokens = self.trunk(tokens)
			v = self.head_v(tokens)
			pi = torch.where(valid_actions, self.head_pi(tokens, self.num_players), self.lowvalue)

		elif self.version == 91:
			tokens = self.embedding(x)
			tokens = self.bilinear_aug(tokens, self.player0_gem_idx)
			tokens = self.trunk(tokens)
			v = self.head_v(tokens)
			pi = torch.where(valid_actions, self.head_pi(tokens, self.num_players), self.lowvalue)

		elif self.version == 92:
			tokens = self.embedding(x)
			tokens = tokens + self.pos_emb
			tokens = self.trunk(tokens)
			v = self.head_v(tokens)
			pi = torch.where(valid_actions, self.head_pi(tokens, self.num_players), self.lowvalue)

		# else:
		# 	raise Exception(f'Unsupported NN version {self.version}')

		return F.log_softmax(pi, dim=1), torch.tanh(v)