import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _make_divisible
from torchvision.models.mobilenetv3 import InvertedResidualConfig, InvertedResidual

from .BotanikConstants import MACHINE_SIZE, NB_ROWS_FOR_MACH
mm = MACHINE_SIZE*MACHINE_SIZE

############################## V11 building blocks #############################

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
		return self.activation(self.norm(result))

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
	def __init__(self, in_channels, exp_channels, out_channels, kernel, use_hs, use_se, setype='avg'):
		super().__init__()
		self.use_res_connect = (in_channels == out_channels)
		activation_layer = nn.Hardswish if use_hs else nn.ReLU

		if exp_channels != in_channels:
			self.expand = LinearNormActivation(in_channels, exp_channels, activation_layer=activation_layer)
		else:
			self.expand = nn.Identity()
		self.depthwise = LinearNormActivation(kernel, kernel, activation_layer=activation_layer, depthwise=True, channels=exp_channels)
		if use_se:
			squeeze_channels = _make_divisible(exp_channels // 4, 8)
			self.se = SqueezeExcitation1d(exp_channels, squeeze_channels, scale_activation=nn.Hardsigmoid, setype=setype)
		else:
			self.se = nn.Identity()
		self.project = LinearNormActivation(exp_channels, out_channels, activation_layer=None)

	def forward(self, input):
		result = self.project(self.se(self.depthwise(self.expand(input))))
		if self.use_res_connect:
			result += input
		return result

def inverted_residual(input_ch, expanded_ch, out_ch, use_se, activation, kernel=3, stride=1, dilation=1, width_mult=1):
	return InvertedResidual(InvertedResidualConfig(input_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation, width_mult), nn.BatchNorm2d)

############################## V12 building blocks #############################

N_CARD_FEAT = 7 + 7 + 2 + 4   # colour one-hot, type one-hot, flowers (1, 3), pipes NESW

class CardFeatures(nn.Module):
	"""Raw int8 card fields (..., 7) -> features (..., 20). Equality ops only: ONNX-safe."""
	def __init__(self):
		super().__init__()
		self.register_buffer('cats', torch.arange(7, dtype=torch.float32))

	def forward(self, x):
		colour = (x[..., 0:1] == self.cats).float()
		ctype  = (x[..., 2:3] == self.cats).float()
		flw    = torch.cat([(x[..., 1:2] == 1).float(), (x[..., 1:2] == 3).float()], dim=-1)
		return torch.cat([colour, ctype, flw, x[..., 3:7]], dim=-1)

class GlobalFeatures(nn.Module):
	"""misc rows -> round, status, main player, scores, remaining cards per colour and per card id."""
	N_FEAT = 1 + 5 + 1 + 3 + 5 + 13 + 1

	def __init__(self):
		super().__init__()
		self.register_buffer('status_cats', torch.arange(5, dtype=torch.float32))
		self.register_buffer('pow2', 2. ** torch.arange(12, -1, -1, dtype=torch.float32))  # card k <-> bit 12-k

	def forward(self, misc):
		rnd    = misc[:, 0, 0:1] / 60.
		status = (misc[:, 0, 1:2] == self.status_cats).float()
		main   = (misc[:, 0, 2:3] == 0).float()
		me, opp = misc[:, 1, 0:1], misc[:, 1, 1:2]
		scores = torch.cat([me / 20., opp / 20., (me - opp) / 10.], dim=-1)
		# Bitfield (rows 3-4, one column per colour): int8 -> unsigned BEFORE any
		# division, since ONNX truncates toward zero where torch floors
		hi, lo = misc[:, 3, 0:5], misc[:, 4, 0:5]
		hi = hi + 256. * (hi < 0).float()
		lo = lo + 256. * (lo < 0).float()
		q = torch.floor((hi * 256. + lo).unsqueeze(-1) / self.pow2)
		bits = q - 2. * torch.floor(q / 2.)                             # (N, 5 colours, 13 ids)
		per_colour, per_id = bits.sum(-1) / 13., bits.sum(1) / 5.
		total = bits.sum((1, 2)).unsqueeze(-1) / 60.
		return torch.cat([rnd, status, main, scores, per_colour, per_id, total], dim=-1)

class ResBlock(nn.Module):
	def __init__(self, c):
		super().__init__()
		self.c1, self.b1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
		self.c2, self.b2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)

	def forward(self, x, bias=None):
		y = F.relu(self.b1(self.c1(x)))
		if bias is not None:
			y = y + bias[:, :, None, None]   # global context injection
		return F.relu(x + self.b2(self.c2(y)))

class TokenBlock(nn.Module):
	"""Pre-norm self-attention + MLP over a handful of tokens."""
	def __init__(self, d, heads):
		super().__init__()
		self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
		self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
		self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

	def forward(self, t):
		h = self.n1(t)
		t = t + self.attn(h, h, h, need_weights=False)[0]
		return t + self.mlp(self.n2(t))

#################################### Network ###################################

class BotanikNNet(nn.Module):
	def __init__(self, game, args):
		super().__init__()
		self.action_size = game.getActionSize()
		self.num_players = game.num_players
		self.args = args
		self.version = args['nn_version']

		if self.version == 11:
			# 1d branch (registers, arrival, freed cards) + one MobileNetV3 branch
			# per machine; each branch outputs full logits, summed at the end
			vect_dim, nb_vect_1d = game.getBoardSize()[-1], 6*5
			self.first_layer_1d = LinearNormActivation(vect_dim, vect_dim, None)
			self.trunk_1d = nn.Sequential(InvertedResidual1d(vect_dim, 3*vect_dim, vect_dim, nb_vect_1d, use_hs=False, use_se=True))
			self.output_layers_PI_1d = nn.Sequential(
				InvertedResidual1d(vect_dim, 3*vect_dim, vect_dim, nb_vect_1d, use_hs=True, use_se=True, setype='max'),
				nn.Flatten(1),
				nn.Linear(vect_dim*nb_vect_1d, self.action_size),
			)
			self.output_layers_V_1d = nn.Sequential(
				InvertedResidual1d(vect_dim, 3*vect_dim, vect_dim, nb_vect_1d, use_hs=True, use_se=True, setype='max'),
				nn.Flatten(1),
				nn.Linear(vect_dim*nb_vect_1d, self.num_players),
			)

			n_filters, n_exp_trunk, n_exp_head, head_depth = 16, 32, 48, 3
			for m in ('mach0', 'mach1'):
				setattr(self, f'first_layer_{m}', nn.Conv2d(7, n_filters, 3, padding=1, bias=False))
				setattr(self, f'trunk_{m}', nn.Sequential(inverted_residual(n_filters, n_exp_trunk, n_filters, False, "RE")))
				setattr(self, f'output_layers_PI_{m}', nn.Sequential(
					*[inverted_residual(n_filters, n_exp_head, n_filters, True, "HS") for _ in range(head_depth)],
					nn.Flatten(1),
					nn.Linear(n_filters*mm, self.action_size),
				))
				setattr(self, f'output_layers_V_{m}', nn.Sequential(
					*[inverted_residual(n_filters, n_exp_head, n_filters, True, "HS") for _ in range(head_depth)],
					nn.Flatten(1),
					nn.Linear(n_filters*mm, self.num_players),
				))

			self.final_layers_PI = nn.Sequential(
				nn.Linear(self.action_size, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size),
			)
			self.final_layers_V = nn.Sequential(
				nn.Linear(self.num_players, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players),
			)

		elif self.version == 12:
			# One-hot card features, conv trunk shared by both machines, token mixer
			# over arrival / columns / freed cards / context, and policy heads aligned
			# with the action layout:
			#   [0:15)  arrival i -> my register slot j   (arrival x column)
			#   [15:30) arrival i -> middle row slot j     (arrival x column)
			#   [30:35) swap mecabot with middle slot j    (column x freed slot 0)
			#   [35:-1) freed card c on cell yx, orient o  (conv head conditioned on c)
			#   [-1]    throw freed cards away             (context)
			C, D = args.get('v12_channels', 32), args.get('v12_dim', 48)
			self.card = CardFeatures()
			self.glob = GlobalFeatures()

			self.stem = nn.Sequential(nn.Conv2d(N_CARD_FEAT, C, 3, padding=1, bias=False), nn.BatchNorm2d(C), nn.ReLU())
			self.cell_emb = nn.Parameter(torch.zeros(1, C, MACHINE_SIZE, MACHINE_SIZE))
			self.res = nn.ModuleList([ResBlock(C) for _ in range(args.get('v12_blocks', 4))])
			self.mach_pool = nn.Linear(2 * C, D)

			self.tok_card = nn.Linear(N_CARD_FEAT, D)
			self.tok_col = nn.Linear(3 * N_CARD_FEAT, D)
			self.tok_glob = nn.Linear(GlobalFeatures.N_FEAT, D)
			self.role = nn.Parameter(0.02 * torch.randn(9, D))  # arrival, column, freed x4, context, my machine, opp machine
			self.blocks = nn.ModuleList([TokenBlock(D, 4) for _ in range(args.get('v12_tblocks', 2))])
			self.norm_out = nn.LayerNorm(D)

			self.ctx_to_mach = nn.Linear(D, C)
			self.pair = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, 2))
			self.swap = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, 1))
			self.freed_to_mach = nn.Linear(D, C)
			self.expand = nn.Sequential(nn.Conv2d(C, C, 1), nn.ReLU(), nn.Conv2d(C, 4, 1))
			self.throw = nn.Linear(D, 1)
			self.value = nn.Sequential(nn.Linear(D + 2 * C, D), nn.ReLU(), nn.Linear(D, self.num_players))

		else:
			raise Exception(f'Unsupported NN version {self.version}')

		self.register_buffer('lowvalue', torch.FloatTensor([-1e8]))

	def forward(self, input_data, valid_actions):
		# input_data is (N, 66, 5, 7); rows after the 2 machines are not for the NN
		x = input_data[:, :(6+2*NB_ROWS_FOR_MACH)]
		x_1d, x_mach0, x_mach1 = x.split([6, NB_ROWS_FOR_MACH, NB_ROWS_FOR_MACH], dim=1)
		x_mach0 = x_mach0.flatten(start_dim=1)[:, :mm*7].reshape(-1, MACHINE_SIZE, MACHINE_SIZE, 7)
		x_mach1 = x_mach1.flatten(start_dim=1)[:, :mm*7].reshape(-1, MACHINE_SIZE, MACHINE_SIZE, 7)

		if self.version == 11:
			x_1d = x_1d.permute(0, 3, 1, 2).flatten(start_dim=2)
			x_1d = F.dropout(self.trunk_1d(self.first_layer_1d(x_1d)), p=self.args['dropout'], training=self.training)
			pi, v = self.output_layers_PI_1d(x_1d), self.output_layers_V_1d(x_1d)
			for m, x_m in (('mach0', x_mach0), ('mach1', x_mach1)):
				x_m = getattr(self, f'trunk_{m}')(getattr(self, f'first_layer_{m}')(x_m.permute(0, 3, 1, 2)))
				pi = pi + getattr(self, f'output_layers_PI_{m}')(x_m)
				v = v + getattr(self, f'output_layers_V_{m}')(x_m)
			v = self.final_layers_V(v)
			pi = torch.where(valid_actions, self.final_layers_PI(pi), self.lowvalue)

		elif self.version == 12:
			N = input_data.shape[0]
			misc, arrival, r_me, r_opp, middle, freed = [x_1d[:, i] for i in range(6)]

			# First half of the trunk on both machines at once
			h = torch.cat([self.card(x_mach0), self.card(x_mach1)], 0).permute(0, 3, 1, 2)
			h = self.stem(h) + self.cell_emb
			half = len(self.res) // 2
			for blk in self.res[:half]:
				h = blk(h)
			t_m = self.mach_pool(torch.cat([h.mean((2, 3)), h.amax((2, 3))], -1))

			r = self.role
			tok = torch.cat([
				self.tok_card(self.card(arrival[:, :3])) + r[0],
				self.tok_col(torch.cat([self.card(middle), self.card(r_me), self.card(r_opp)], -1)) + r[1],
				self.tok_card(self.card(freed[:, :4])) + r[2:6],
				(self.tok_glob(self.glob(misc)) + r[6]).unsqueeze(1),
				(t_m[:N] + r[7]).unsqueeze(1),
				(t_m[N:] + r[8]).unsqueeze(1),
			], dim=1)
			for blk in self.blocks:
				tok = blk(tok)
			tok = self.norm_out(tok)
			t_arr, t_col, t_fr, t_g = tok[:, 0:3], tok[:, 3:8], tok[:, 8:12], tok[:, 12]

			# Second half of the trunk: my machine only, with context injected
			h_me, h_op = h[:N], h[N:]
			ctx = self.ctx_to_mach(t_g)
			for blk in self.res[half:]:
				h_me = blk(h_me, ctx)

			pair = self.pair(torch.cat([t_arr.unsqueeze(2).expand(-1, -1, 5, -1), t_col.unsqueeze(1).expand(-1, 3, -1, -1)], -1))  # (N, 3, 5, 2)
			swap = self.swap(torch.cat([t_col, t_fr[:, 0:1].expand(-1, 5, -1)], -1)).squeeze(-1)                                 # (N, 5)
			cond = self.freed_to_mach(t_fr[:, 0:2])                                                                               # (N, 2, C)
			e = self.expand((h_me.unsqueeze(1) + cond[:, :, :, None, None]).flatten(0, 1))                                       # (2N, 4, M, M)
			e = e.reshape(N, 2, 4, MACHINE_SIZE, MACHINE_SIZE).permute(0, 1, 3, 4, 2).flatten(1)                                 # (N, 8mm)
			logits = torch.cat([pair[..., 0].flatten(1), pair[..., 1].flatten(1), swap, e, self.throw(t_g)], -1)
			pi = torch.where(valid_actions, logits, self.lowvalue)
			v = self.value(torch.cat([t_g, h_me.mean((2, 3)), h_op.mean((2, 3))], -1))

		return F.log_softmax(pi, dim=1), torch.tanh(v)

# Checkpoints written while V12 had its own class pickle 'full_model' under this name
BotanikNNetV12 = BotanikNNet
