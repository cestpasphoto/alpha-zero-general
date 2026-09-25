import torch
import torch.nn as nn
import torch.nn.functional as F

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
		# These columns hold int8 BYTES: a byte >= 128 is stored as a NEGATIVE int8.
		# % 256 first makes the operand non-negative, so that torch (floor) and the
		# exported ONNX graph (truncation) extract the same bits.
		bitfield3 = (x[..., 3].long() % 256).unsqueeze(-1)
		bitfield4 = (x[..., 4].long() % 256).unsqueeze(-1)
		
		# Integer division by powers of 2, then modulo 2
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
	Everything starts at exactly zero: at step 0 the network computes the same
	function as without this module, training decides how much of it to use.
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

		# Static tables, persistent=False: not written into state_dict
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
		# g_ctx is permutation-invariant over deck rows, so global_head alone
		# cannot express "slot i holds this combo".
		self.choose_head = nn.Linear(d_model, 1)

	def zero_init_additive(self):
		nn.init.zeros_(self.choose_head.weight)
		nn.init.zeros_(self.choose_head.bias)

	def forward(self, tokens):
		# tokens: (Batch, nb_vect, D)
		local_tokens  = tokens[:, :self.nb_areas, :]
		global_tokens = tokens[:, self.nb_areas:, :]
		deck_tokens   = global_tokens[:, self.deck_start:self.deck_start + self.deck_size, :]
		
		# Local logits: (Batch, nb_areas, 5)
		l_logits = self.local_head(local_tokens)
		
		# Global context & logits: (Batch, 16)
		g_ctx = global_tokens.mean(dim=1)
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
	def __init__(self, game, args):
		super(SmallworldNNet, self).__init__()
		self.nb_vect, self.vect_dim = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.args = args
		self.version = args['nn_version']
		self.register_buffer('lowvalue', torch.FloatTensor([-1e8]))

		if self.version == 62: # Self-attention transformer, D=48
			D = 48
			self.stem = InputStem(d_model=D)
			self.head = ActionSlicerHead(d_model=D, action_size=self.action_size, num_players=self.num_players)
			
			encoder_layer = nn.TransformerEncoderLayer(
				d_model=D, nhead=3, dim_feedforward=D*4, 
				dropout=self.args.get('dropout', 0.1), batch_first=True
			)
			self.trunk = nn.TransformerEncoder(encoder_layer, num_layers=3)
			# Static map / token-identity injection (position, terrain, adjacency),
			# additive and zero-initialised: see StaticMapEmbedding
			self.map_emb = StaticMapEmbedding(D, self.nb_vect, self.head.nb_areas)
		else:
			raise Exception(f'Unsupported NN version {self.version}')

		self.apply(self._init_weights)
		# MUST come after self.apply(), which would re-randomise the additive modules
		self.map_emb.zero_init()
		self.head.zero_init_additive()

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			nn.init.kaiming_uniform_(m.weight)
			if m.bias is not None:
				nn.init.zeros_(m.bias)

	def forward(self, input_data, valid_actions):
		# input_data is (N, H, C) typically (N, 40, 8)
		x = input_data.view(-1, self.nb_vect, self.vect_dim)
		x = self.stem(x)
		# Token identity + terrain, added AFTER the stem LayerNorm (zero at init)
		x = x + self.map_emb().unsqueeze(0)

		if self.training and self.args.get('dropout', 0) > 0:
			x = F.dropout(x, p=self.args['dropout'])

		# Adjacency as an additive attention bias in every trunk layer
		# (float mask = added to the attention logits; all-zero at init)
		x = self.trunk(x, mask=self.map_emb.attention_bias())
		pi, v = self.head(x)

		# Mask invalid actions
		pi = torch.where(valid_actions, pi, self.lowvalue)
		return F.log_softmax(pi, dim=1), torch.tanh(v)
