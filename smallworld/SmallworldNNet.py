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

class InputStem(nn.Module):
	"""
	Parses the heterogeneous (N, 8) input into clean tokens of size D.
	Handles explicitly the offset of negative declined values and bitfield extraction.
	"""
	def __init__(self, d_model):
		super().__init__()
		# Embeddings for categorical features
		self.emb_ppl    = nn.Embedding(31, d_model)  # Types from -15 to +15
		self.emb_pwr    = nn.Embedding(41, d_model)  # Powers from -20 to +20
		self.emb_player = nn.Embedding(6, d_model)   # Player IDs from -1 to 4
		
		# Linear projections
		self.num_proj = nn.Linear(5, d_model)        # For continuous numericals
		self.bit_proj = nn.Linear(16, d_model)       # For unpacked bitfields
		
		# Final mix
		self.out_proj = nn.Linear(d_model * 5, d_model)
		self.norm     = nn.LayerNorm(d_model)

		self.register_buffer('powers_of_2', torch.tensor([2**i for i in range(8)], dtype=torch.long))

	def forward(self, x):
		# x shape: (Batch, nb_vect, 8)
		
		# 1. Categorical Embeddings (with safe offsets)
		ppl_type = torch.clamp(x[..., 1] + 15, 0, 30).long()
		power    = torch.clamp(x[..., 2] + 20, 0, 40).long()
		player   = torch.clamp(x[..., 7] + 1,  0, 5).long()
		
		e_ppl    = self.emb_ppl(ppl_type)
		e_pwr    = self.emb_pwr(power)
		e_player = self.emb_player(player)
		
		# 2. Continuous numericals (cols: 0, 3, 4, 5, 6)
		numericals = torch.stack([x[..., 0], x[..., 3], x[..., 4], x[..., 5], x[..., 6]], dim=-1)
		e_num = self.num_proj(numericals.float() / 10.0) # Soft normalization
		
		# 3. Explicit Bitfield Extraction for cols 3 and 4 (8 bits each)
		# These columns hold int8 BYTES (invisible_deck packbits, sorcerer /
		# diplomat bitfields): a byte >= 128 is stored as a NEGATIVE int8.
		# Torch '//' floors, so it already extracts the correct unsigned bits --
		# but ONNX integer division TRUNCATES towards zero, so the exported graph
		# (used by MCTS at play time) computes different bits than training does.
		# Taking % 256 first makes the operand non-negative, where floor == trunc:
		# identical torch output, ONNX finally agreeing with it.
		bitfield3 = (x[..., 3].long() % 256).unsqueeze(-1)
		bitfield4 = (x[..., 4].long() % 256).unsqueeze(-1)
		
		# Division entière par les puissances de 2, puis modulo 2
		bits3 = (bitfield3 // self.powers_of_2) % 2
		bits4 = (bitfield4 // self.powers_of_2) % 2
		
		e_bits = self.bit_proj(torch.cat([bits3.float(), bits4.float()], dim=-1))
		
		# 4. Concatenate and project
		concat_feat = torch.cat([e_ppl, e_pwr, e_player, e_num, e_bits], dim=-1)
		tokens = self.out_proj(concat_feat)
		return self.norm(tokens)

class StaticMapEmbedding(nn.Module):
	"""
	Additive, ZERO-INITIALISED injection of the information the state tensor
	does not carry (see SmallworldLogicNumba.py header):
	  * token identity   : learned positional embedding over ALL nb_vect rows
	                       (area id for territories, slot id = cost for deck rows,
	                        player slot for status rows);
	  * terrain          : embedding of descr[:,0] + linear on descr[:,1:6]
	                       (cavern, magic, mine, lost-tribe-at-start, at-edge),
	                       area rows only;
	  * adjacency        : additive attention bias alpha * connexity_matrix on
	                       area x area entries, alpha learned, alpha = 0 at init.
	Everything starts at exactly zero, so a network loaded from a checkpoint
	that predates this module computes EXACTLY the same function as before:
	trunk, stem and heads are untouched and the champion's Elo is preserved at
	step 0. Training then decides how much of it to use.
	"""
	def __init__(self, d_model, nb_vect, nb_areas):
		super().__init__()
		# Import here: NUMBER_PLAYERS must be resolved when the net is built,
		# not at module import time.
		from .SmallworldMaps import connexity_matrix, descr
		assert descr.shape[0] == nb_areas, f'descr rows {descr.shape[0]} != nb_areas {nb_areas}'
		assert connexity_matrix.shape == (nb_areas, nb_areas)

		self.nb_vect, self.nb_areas = nb_vect, nb_areas
		self.pos_emb      = nn.Embedding(nb_vect, d_model)      # token identity, all rows
		self.terrain_emb  = nn.Embedding(6, d_model)            # FORESTT..WATER, area rows
		self.terrain_flag = nn.Linear(5, d_model, bias=False)   # cavern/magic/mine/lost-tribe/edge
		self.adj_alpha    = nn.Parameter(torch.zeros(()))       # attention bias strength

		# Static tables. persistent=False: not written into state_dict, so an old
		# and a new checkpoint differ ONLY by the learnable tensors listed above.
		self.register_buffer('token_idx', torch.arange(nb_vect), persistent=False)
		self.register_buffer('terrain_id', torch.as_tensor(descr[:, 0]).long(), persistent=False)
		self.register_buffer('terrain_flags', torch.as_tensor(descr[:, 1:6]).float(), persistent=False)
		bias = torch.zeros(nb_vect, nb_vect)
		bias[:nb_areas, :nb_areas] = torch.as_tensor(connexity_matrix).float()   # 0/1, zero diagonal
		self.register_buffer('adj_bias', bias, persistent=False)

		self.zero_init()

	def zero_init(self):
		nn.init.zeros_(self.pos_emb.weight)
		nn.init.zeros_(self.terrain_emb.weight)
		nn.init.zeros_(self.terrain_flag.weight)
		nn.init.zeros_(self.adj_alpha)

	def forward(self):
		# Returns (nb_vect, D), to be broadcast-added to the stem tokens.
		feat = self.pos_emb(self.token_idx)
		terr = self.terrain_emb(self.terrain_id) + self.terrain_flag(self.terrain_flags)
		return torch.cat([feat[:self.nb_areas] + terr, feat[self.nb_areas:]], dim=0)

	def attention_bias(self):
		# (nb_vect, nb_vect) float mask ADDED to every attention logit of the
		# trunk (nn.TransformerEncoder(..., mask=...)). All-zero at init.
		return self.adj_alpha * self.adj_bias


class GraphMixer(nn.Module):
	"""
	GROWTH MODULE 1 -- adjacency, done properly this time.

	The scalar `StaticMapEmbedding.adj_alpha` never left zero (6.9e-03 even at
	lr x30): ONE parameter shared by 3 layers x 3 heads receives gradients whose
	sign disagrees between heads, and they cancel. That is a design flaw of the
	patch, not evidence that adjacency is useless.

	Here each message-passing step has its own D x D matrices, so the gradient is
	a matrix, not a single averaged scalar. Message passing runs on AREA tokens
	only, before the trunk, as a residual branch:
	    area <- area + out(relu(ln(self(area) + neigh(A_hat @ area))))
	`out` is zero-initialised, so the branch contributes exactly nothing at step
	0 and the champion's function is preserved. `self`/`neigh` are randomly
	initialised, so `out` sees a non-zero input and receives gradient from the
	very first batch (the usual ReZero / LoRA-B ordering: the gate opens first,
	the inner weights follow).

	Cost at D=48, A=30, 2 layers: ~0.36 MFLOPs against ~4.3 for the trunk (+8%).
	"""
	def __init__(self, d_model, nb_areas, n_layers=2):
		super().__init__()
		from .SmallworldMaps import connexity_matrix
		assert connexity_matrix.shape == (nb_areas, nb_areas)
		a = torch.as_tensor(connexity_matrix).float() + torch.eye(nb_areas)
		deg = a.sum(-1).clamp(min=1.0).pow(-0.5)
		self.register_buffer('adj_norm', deg.unsqueeze(1) * a * deg.unsqueeze(0), persistent=False)

		self.nb_areas = nb_areas
		self.self_proj  = nn.ModuleList(nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers))
		self.neigh_proj = nn.ModuleList(nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers))
		self.ln         = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
		self.out_proj   = nn.ModuleList(nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers))
		self.zero_init()

	def zero_init(self):
		# ONLY the output gate. Zeroing the inner layers too would starve the gate
		# of gradient (its own gradient is proportional to the branch activation).
		for m in self.out_proj:
			nn.init.zeros_(m.weight)

	def forward(self, area):
		# area: (B, A, D) -- returns the same shape, identity at init.
		for s_, n_, ln_, o_ in zip(self.self_proj, self.neigh_proj, self.ln, self.out_proj):
			neigh = torch.einsum('ij,bjd->bid', self.adj_norm, area)
			area = area + o_(F.relu(ln_(s_(area) + n_(neigh))))
		return area


class AttentionPool(nn.Module):
	"""
	GROWTH MODULE 2 -- replace mean() pooling by a learned weighted pooling,
	starting from EXACTLY the mean.

	`ActionSlicerHead` reads the value and all global actions from
	`global_tokens.mean(dim=1)`: 22 heterogeneous rows in 3 players (9 peoples,
	6 deck, 3 round_status, 3 game_status, 1 invisible_deck) averaged into one
	vector, which is also the reason the Choose logits could not see slot
	identity (F3). A mean cannot down-weight an irrelevant row.

	Attention logits are  (W_k(tokens) . q) / sqrt(D)  with the query q
	initialised to ZERO: every logit is 0, softmax is uniform, the output is the
	mean, bit for bit. q receives gradient immediately (W_k(tokens) != 0); W_k
	follows once q has moved. No value projection, so the pooled vector stays in
	the space value_head/global_head already read.
	"""
	def __init__(self, d_model):
		super().__init__()
		self.scale = d_model ** -0.5
		self.key   = nn.Linear(d_model, d_model)
		self.query = nn.Parameter(torch.zeros(d_model))
		self.zero_init()

	def zero_init(self):
		nn.init.zeros_(self.query)

	def forward(self, tokens):
		# tokens: (B, G, D) -> (B, D). BIT-EXACTLY tokens.mean(1) while query == 0.
		# Written as mean + deviation on purpose: a plain weighted sum with
		# uniform weights differs from mean() by float rounding (~1e-7), which
		# would break the "same function at step 0" guarantee for no reason.
		# With zero logits softmax returns exactly 1/G, so the deviation is
		# exactly 0 and the correction term vanishes.
		logits = (self.key(tokens) @ self.query) * self.scale       # (B, G)
		w = torch.softmax(logits, dim=1) - 1.0 / tokens.size(1)     # (B, G)
		return tokens.mean(dim=1) + (w.unsqueeze(-1) * tokens).sum(dim=1)


def _make_identity_layer(d_model, nhead, dropout):
	"""
	GROWTH MODULE 3 -- a 4th transformer layer that is the IDENTITY at init.

	PRE-norm is mandatory here. The trunk's layers are post-norm
	(x <- LayerNorm(x + sublayer(x))), so zeroing the sublayers still leaves a
	LayerNorm on the residual stream: NOT identity. With norm_first=True the
	block is x <- x + sublayer(norm(x)), so zeroing the two output projections
	(attention out_proj and the second FF linear) makes it exactly identity.
	The extra layer being pre-norm while the trunk is post-norm is fine: it is a
	residual block appended after the trunk, not inserted inside it.
	"""
	layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
	                                   dropout=dropout, batch_first=True, norm_first=True)
	nn.init.zeros_(layer.self_attn.out_proj.weight)
	nn.init.zeros_(layer.self_attn.out_proj.bias)
	nn.init.zeros_(layer.linear2.weight)
	nn.init.zeros_(layer.linear2.bias)
	return layer


class ActionSlicerHead(nn.Module):
	"""
	Reconstructs the 1D action vector without Flattening, maintaining spatial 
	integrity for local actions and pooling context for global actions.
	"""
	def __init__(self, d_model, action_size, num_players):
		super().__init__()
		# Deduce nb_areas dynamically based on action_size formula:
		# action_size = 5 * nb_areas + 8 (RedeployN) + 6 (Deck) + 1 (Decline) + 1 (End)
		self.nb_areas = (action_size - 16) // 5
		# Deck rows sit right after the 3*P people rows (see Board.copy_state()).
		self.deck_start = 3 * num_players
		self.deck_size = 6
		
		self.local_head  = nn.Linear(d_model, 5)  # 5 actions per area
		self.global_head = nn.Linear(d_model, 16) # 16 global actions
		self.value_head  = nn.Linear(d_model, num_players)

		# Additive, zero-initialised: per-deck-token Choose logit. The pooled
		# g_ctx is permutation-invariant over deck rows, and no row carries its
		# own index, so global_head alone CANNOT express "slot i holds this
		# combo" -- it only sees the multiset of the 6 combos. This head adds a
		# per-slot term on top of the existing pooled logit.
		self.choose_head = nn.Linear(d_model, 1)
		# Optional: learned pooling instead of mean(). Identical to mean at init.
		self.pool = None

	def enable_attention_pool(self, d_model):
		self.pool = AttentionPool(d_model)

	def zero_init_additive(self):
		nn.init.zeros_(self.choose_head.weight)
		nn.init.zeros_(self.choose_head.bias)
		if getattr(self, 'pool', None) is not None:
			self.pool.zero_init()

	def forward(self, tokens):
		# tokens: (Batch, nb_vect, D)
		local_tokens  = tokens[:, :self.nb_areas, :]
		global_tokens = tokens[:, self.nb_areas:, :]
		deck_tokens   = global_tokens[:, self.deck_start:self.deck_start + self.deck_size, :]
		
		# Local logits: (Batch, nb_areas, 5)
		l_logits = self.local_head(local_tokens)
		
		# Global context & logits: (Batch, 16)
		pool = getattr(self, 'pool', None)
		g_ctx = pool(global_tokens) if pool is not None else global_tokens.mean(dim=1)
		g_logits = self.global_head(g_ctx)
		choose = g_logits[:, 8:14] + self.choose_head(deck_tokens).squeeze(-1)   # (Batch, 6)
		
		# Reconstruct exactly matching valid_moves layout
		pi = torch.cat([
			l_logits[..., 0],      # 0 to A-1: Abandon
			l_logits[..., 1],      # A to 2A-1: Attack
			l_logits[..., 2],      # 2A to 3A-1: SpecPpl
			l_logits[..., 3],      # 3A to 4A-1: SpecPwr
			g_logits[:, 0:8],      # Redeploy N (MAX_REDEPLOY=8)
			l_logits[..., 4],      # Redeploy 1 (NB_AREAS)
			choose,                # Choose (DECK_SIZE=6) = pooled + per-slot
			g_logits[:, 14:15],    # Decline (1)
			g_logits[:, 15:16]     # End (1)
		], dim=1)
		
		v = self.value_head(g_ctx)
		return pi, v

class SmallworldNNet(nn.Module):
	# Input-encoding contract, checked by GenericNNetWrapper.load_network: weights
	# trained before the % 256 fix encode a different function (see F4).
	bit_semantics = 'unsigned'
	use_features = False
	use_adj_bias = True
	additive_param_prefixes = ()

	def __init__(self, game, args):
		super(SmallworldNNet, self).__init__()
		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.args = args
		self.version = args['nn_version']
		self.register_buffer('lowvalue', torch.FloatTensor([-1e8]))
			
		if self.version == 31: # Like V21 but in bigger

			# Some input data are categorical so one track for improvement
			# would be to use nn.Embedding()

			self.first_layer = LinearNormActivation(self.nb_vect, 256, None)
			confs  = []
			confs += [InvertedResidual1d(256, 384, 256, 8, False, "RE")]
			confs += [InvertedResidual1d(256, 384, 256, 8, False, "RE")]
			confs += [InvertedResidual1d(256, 384, 256, 8, False, "RE")]
			confs += [InvertedResidual1d(256, 384, 256, 8, False, "RE")]
			self.trunk = nn.Sequential(*confs)

			n_filters = 128
			head_PI = [
				InvertedResidual1d(256, 384, 256, 8, True, "HS", setype='avg'),
				InvertedResidual1d(256, 384, 256, 8, True, "HS", setype='max'),
				InvertedResidual1d(256, 384, 256, 8, True, "HS", setype='max'),
				InvertedResidual1d(256, 384, 128, 8, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, n_filters *8),
				nn.ReLU(),
				nn.Linear(n_filters *8, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [
				InvertedResidual1d(256, 384, 256, 8, True, "HS", setype='avg'),
				InvertedResidual1d(256, 384, 256, 8, True, "HS", setype='max'),
				InvertedResidual1d(256, 384, 256, 8, True, "HS", setype='max'),
				InvertedResidual1d(256, 384, 128, 8, True, "HS", setype='max'),
				nn.Flatten(1),
				nn.Linear(n_filters *8, n_filters *8),
				nn.ReLU(),
				nn.Linear(n_filters *8, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			]
			self.output_layers_V = nn.Sequential(*head_V)

		# ARCHITECTURE V42 : Full Self-Attention Transformer
		elif self.version == 42:
			D = 64
			self.stem = InputStem(d_model=D)
			self.head = ActionSlicerHead(d_model=D, action_size=self.action_size, num_players=self.num_players)

			encoder_layer = nn.TransformerEncoderLayer(
				d_model=D, nhead=4, dim_feedforward=D*4, 
				dropout=self.args.get('dropout', 0.1), batch_first=True
			)
			self.trunk = nn.TransformerEncoder(encoder_layer, num_layers=3)

		elif self.version == 62: # same as 42 but even SMALLER
			D = 48
			self.stem = InputStem(d_model=D)
			self.head = ActionSlicerHead(d_model=D, action_size=self.action_size, num_players=self.num_players)
			
			encoder_layer = nn.TransformerEncoderLayer(
				d_model=D, nhead=3, dim_feedforward=D*4, 
				dropout=self.args.get('dropout', 0.1), batch_first=True
			)
			self.trunk = nn.TransformerEncoder(encoder_layer, num_layers=3)

		# else:
		# 	raise Exception(f'Unsupported NN version {self.version}')

		if self.version in [42, 62]:
			# Static map / token-identity injection (position, terrain, adjacency).
			# Additive and zero-initialised: see StaticMapEmbedding docstring.
			self.map_emb = StaticMapEmbedding(D, self.nb_vect, self.head.nb_areas)
			self.use_adj_bias = bool(self.args.get('adj_bias', True))
			# Rules-derived per-area features (conquest cost, points if conquered,
			# frontier degree...). Deterministic function of the SAME 8-column
			# state, so examples and checkpoints stay valid. Off by default;
			# projection is zero-init, so enabling it does not change the
			# function at step 0 either.
			self.use_features = bool(self.args.get('map_features', False))
			if self.use_features:
				from .SmallworldFeatures import MapFeatureBlock
				self.feat = MapFeatureBlock(self.num_players)
				self.feat_proj = nn.Linear(MapFeatureBlock.K, D, bias=False)
			# Tensors that may legitimately be MISSING from an older checkpoint.
			# GenericNNetWrapper.load_network accepts a checkpoint whose missing
			# keys ALL start with one of these prefixes (and which has no
			# unexpected key and no shape mismatch); anything else is refused.
			# ---- growth modules (all OFF by default, all function-preserving) ----
			# Each one is a strict addition whose output gate starts at zero, so a
			# net warm-started from a checkpoint that lacks them computes exactly
			# the same function at step 0. Enable ONE per experimental arm.
			self.use_graph_mix = bool(self.args.get('graph_mix', False))
			if self.use_graph_mix:
				self.graph_mix = GraphMixer(D, self.head.nb_areas,
				                            n_layers=int(self.args.get('graph_layers', 2)))
			self.use_attn_pool = bool(self.args.get('attn_pool', False))
			if self.use_attn_pool:
				self.head.enable_attention_pool(D)
			# GROWTH MODULE 4 -- give the VALUE head access to the AREA tokens.
			# Measured motivation: on a clean by-game held-out, a linear probe on
			# g_ctx explains ~5% of the outcome over rounds 1-4 (42% of positions)
			# and the champion's OWN value head scores the same (VE_champ ~= the
			# probe). The head is already saturated on its input, so more head
			# capacity cannot help -- but its input is the 22 pooled GLOBAL rows
			# only, while the early-game decision state IS the map. This adds a
			# pooled read of the 30 area tokens, zero-gated, so v is unchanged at
			# step 0. If rounds 1-4 do not move, the map is not the missing input.
			self.use_area_value = bool(self.args.get('area_value', False))
			if self.use_area_value:
				self.area_pool = AttentionPool(D)
				self.area_value = nn.Sequential(nn.Linear(D, D), nn.ReLU(),
				                                nn.Linear(D, self.num_players))
			self.use_extra_layer = bool(self.args.get('extra_layer', False))
			if self.use_extra_layer:
				self.extra_layer = _make_identity_layer(D, 3 if self.version == 62 else 4,
				                                        self.args.get('dropout', 0.1))

			# Tensors that may legitimately be MISSING from an older checkpoint.
			# GenericNNetWrapper.load_network accepts a checkpoint whose missing
			# keys ALL start with one of these prefixes (and which has no
			# unexpected key and no shape mismatch); anything else is refused.
			self.additive_param_prefixes = ('map_emb.', 'head.choose_head.', 'feat_proj.',
			                                'graph_mix.', 'head.pool.', 'extra_layer.',
			                                'area_pool.', 'area_value.')

		self.apply(self._init_weights)

		# MUST come after self.apply(): kaiming would otherwise re-randomise the
		# additive modules and break the "same function as the champion" guarantee.
		if self.version in [42, 62]:
			self.map_emb.zero_init()
			self.head.zero_init_additive()
			if self.use_features:
				nn.init.zeros_(self.feat_proj.weight)
			if self.use_graph_mix:
				self.graph_mix.zero_init()
			if self.use_area_value:
				# Only the LAST linear: zeroing the first one too would starve the
				# gate of gradient. The pooling query follows once the gate opens.
				self.area_pool.zero_init()
				nn.init.zeros_(self.area_value[-1].weight)
				nn.init.zeros_(self.area_value[-1].bias)
			if self.use_extra_layer:
				# _make_identity_layer already zeroed them; apply() re-randomised.
				nn.init.zeros_(self.extra_layer.self_attn.out_proj.weight)
				nn.init.zeros_(self.extra_layer.self_attn.out_proj.bias)
				nn.init.zeros_(self.extra_layer.linear2.weight)
				nn.init.zeros_(self.extra_layer.linear2.bias)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			nn.init.kaiming_uniform_(m.weight)
			if m.bias is not None:
				nn.init.zeros_(m.bias)

	def upgrade_legacy(self):
		"""
		Repair an instance restored from checkpoint['full_model'] (pickled by the
		code that predates the additive modules): pickle restores the old
		__dict__ but the METHODS come from the current class, so forward() would
		look for submodules that this instance never had.

		Adds them, zero-initialised, on the right device. The module then
		computes EXACTLY what the pickled one computed. Idempotent.
		"""
		if self.version not in [42, 62]:
			return self
		device = next(self.parameters()).device
		D = self.stem.out_proj.out_features
		if not hasattr(self.head, 'pool'):
			self.head.pool = None
		for flag in ('use_graph_mix', 'use_attn_pool', 'use_extra_layer', 'use_area_value'):
			if not hasattr(self, flag):
				setattr(self, flag, False)
		if not hasattr(self, 'use_features'):
			self.use_features = False
		if not hasattr(self.head, 'choose_head'):
			self.head.deck_start = 3 * self.num_players
			self.head.deck_size = 6
			self.head.choose_head = nn.Linear(D, 1).to(device)
			self.head.zero_init_additive()
		if not hasattr(self, 'map_emb'):
			self.map_emb = StaticMapEmbedding(D, self.nb_vect, self.head.nb_areas).to(device)
			self.map_emb.zero_init()
		self.use_adj_bias = bool(self.args.get('adj_bias', True))
		self.additive_param_prefixes = ('map_emb.', 'head.choose_head.', 'feat_proj.',
		                                'graph_mix.', 'head.pool.', 'extra_layer.',
		                                'area_pool.', 'area_value.')
		return self

	def forward(self, input_data, valid_actions):
		# input_data is (N, H, C) typically (N, 40, 8)
		if self.version in [31]: # Use input as is
			x = input_data.view(-1, self.nb_vect, self.vect_dim) # no transpose
			x = self.first_layer(x)
			x = F.dropout(self.trunk(x), p=self.args['dropout'], training=self.training)
			v = self.output_layers_V(x)
			pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)

		elif self.version in [42, 62]:
			x = input_data.view(-1, self.nb_vect, self.vect_dim)
			raw = x
			x = self.stem(x)
			# Token identity + terrain, added AFTER the stem LayerNorm (zero at init)
			x = x + self.map_emb().unsqueeze(0)
			if self.use_features:
				# Rules-derived features on AREA tokens only (zero at init)
				A = self.head.nb_areas
				f = self.feat_proj(self.feat(raw))
				x = x + F.pad(f, (0, 0, 0, self.nb_vect - A))
			if self.use_graph_mix:
				# Message passing on AREA tokens only (identity at init)
				A = self.head.nb_areas
				x = torch.cat([self.graph_mix(x[:, :A]), x[:, A:]], dim=1)
			
			# Run trunk with optional dropout if needed
			if self.training and self.args.get('dropout', 0) > 0 and self.version != 42:
				x = F.dropout(x, p=self.args['dropout'])
				
			# Adjacency as an additive attention bias in every trunk layer
			# (float mask = added to the attention logits; all-zero at init)
			attn_bias = self.map_emb.attention_bias() if self.use_adj_bias else None
			x = self.trunk(x, mask=attn_bias)
			if self.use_extra_layer:
				x = self.extra_layer(x, src_mask=attn_bias)
			pi, v = self.head(x)
			if getattr(self, 'use_area_value', False):
				# Pooled read of the AREA tokens, added BEFORE the tanh so the
				# module is exactly the identity while its gate is zero.
				A = self.head.nb_areas
				v = v + self.area_value(self.area_pool(x[:, :A]))
			
			# Mask invalid actions
			pi = torch.where(valid_actions, pi, self.lowvalue)
			
		else:
			raise Exception(f'Unsupported NN version {self.version}')

		return F.log_softmax(pi, dim=1), torch.tanh(v)

