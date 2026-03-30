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
		bitfield3 = x[..., 3].long().unsqueeze(-1)
		bitfield4 = x[..., 4].long().unsqueeze(-1)
		
		# Division entière par les puissances de 2, puis modulo 2
		bits3 = (bitfield3 // self.powers_of_2) % 2
		bits4 = (bitfield4 // self.powers_of_2) % 2
		
		e_bits = self.bit_proj(torch.cat([bits3.float(), bits4.float()], dim=-1))
		
		# 4. Concatenate and project
		concat_feat = torch.cat([e_ppl, e_pwr, e_player, e_num, e_bits], dim=-1)
		tokens = self.out_proj(concat_feat)
		return self.norm(tokens)

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
		
		self.local_head  = nn.Linear(d_model, 5)  # 5 actions per area
		self.global_head = nn.Linear(d_model, 16) # 16 global actions
		self.value_head  = nn.Linear(d_model, num_players)

	def forward(self, tokens):
		# tokens: (Batch, nb_vect, D)
		local_tokens  = tokens[:, :self.nb_areas, :]
		global_tokens = tokens[:, self.nb_areas:, :]
		
		# Local logits: (Batch, nb_areas, 5)
		l_logits = self.local_head(local_tokens)
		
		# Global context & logits: (Batch, 16)
		g_ctx = global_tokens.mean(dim=1)
		g_logits = self.global_head(g_ctx)
		
		# Reconstruct exactly matching valid_moves layout
		pi = torch.cat([
			l_logits[..., 0],      # 0 to A-1: Abandon
			l_logits[..., 1],      # A to 2A-1: Attack
			l_logits[..., 2],      # 2A to 3A-1: SpecPpl
			l_logits[..., 3],      # 3A to 4A-1: SpecPwr
			g_logits[:, 0:8],      # Redeploy N (MAX_REDEPLOY=8)
			l_logits[..., 4],      # Redeploy 1 (NB_AREAS)
			g_logits[:, 8:14],     # Choose (DECK_SIZE=6)
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

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			nn.init.kaiming_uniform_(m.weight)
			if m.bias is not None:
				nn.init.zeros_(m.bias)

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
			x = self.stem(x)
			
			# Run trunk with optional dropout if needed
			if self.training and self.args.get('dropout', 0) > 0 and self.version != 42:
				x = F.dropout(x, p=self.args['dropout'])
				
			x = self.trunk(x)
			pi, v = self.head(x)
			
			# Mask invalid actions
			pi = torch.where(valid_actions, pi, self.lowvalue)
			
		else:
			raise Exception(f'Unsupported NN version {self.version}')

		return F.log_softmax(pi, dim=1), torch.tanh(v)

