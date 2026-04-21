import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import InvertedResidualConfig, InvertedResidual

# =========================================================================
# 1. BUILDING BLOCKS
# =========================================================================

class SimpleResBlock(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(channels)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(channels)

	def forward(self, x):
		residual = x
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += residual
		return F.relu(out)

# =========================================================================
# 2. MAIN NETWORK ARCHITECTURE
# =========================================================================

class AbaloneNNet(nn.Module):
	def __init__(self, game, args):
		self.board_size = game.getBoardSize()   # (9, 9, 4)
		self.action_size = game.getActionSize() # 3402 (9 * 9 * 42)
		self.num_players = 2
		self.args = args
		self.version = args['nn_version']
		
		super(AbaloneNNet, self).__init__()

		# V10: Standard ResNet architecture (~1.5 MFlops)
		if self.version == 10:
			n_filters = 24
			depth = 4
			
			self.first_layer = nn.Sequential(
				nn.Conv2d(3, n_filters, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(n_filters),
				nn.ReLU()
			)
			self.trunk = nn.Sequential(*[SimpleResBlock(n_filters) for _ in range(depth)])
			
			# Metadata processing (P0 score, P1 score, round)
			self.meta_fc = nn.Sequential(
				nn.Linear(3, 16),
				nn.ReLU()
			)
			
			# Policy Head
			self.head_PI = nn.Sequential(
				nn.Conv2d(n_filters, 42, kernel_size=1, bias=False),
				nn.BatchNorm2d(42)
			)
			
			# Value Head (Spatial pooling + Meta data fusion)
			self.head_V_conv = nn.Sequential(
				nn.Conv2d(n_filters, 4, kernel_size=1, bias=False),
				nn.BatchNorm2d(4),
				nn.ReLU()
			)
			self.head_V_fc = nn.Sequential(
				nn.Linear(4 * 9 * 9 + 16, 64),
				nn.ReLU(),
				nn.Linear(64, self.num_players)
			)

		# V20: MobileNetV3 / Inverted Residuals architecture (~3 MFlops)
		elif self.version == 20:
			n_filters = 32
			depth = 5
			
			def inverted_residual(input_ch, expanded_ch, out_ch, use_se, activation, kernel=3, stride=1, dilation=1, width_mult=1):
				return InvertedResidual(InvertedResidualConfig(input_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation, width_mult), nn.BatchNorm2d)
			
			self.first_layer = nn.Sequential(
				nn.Conv2d(3, n_filters, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(n_filters),
				nn.ReLU()
			)
			
			# Expansion factor of 3 (n_filters * 3 = 96 internal channels)
			confs = [inverted_residual(n_filters, n_filters*3, n_filters, False, "RE") for _ in range(depth)]
			self.trunk = nn.Sequential(*confs)
			
			# Metadata processing
			self.meta_fc = nn.Sequential(
				nn.Linear(3, 16),
				nn.ReLU()
			)

			# Policy Head
			self.head_PI = nn.Sequential(
				nn.Conv2d(n_filters, 42, kernel_size=1, bias=False),
				nn.BatchNorm2d(42)
			)
			
			# Value Head
			self.head_V_conv = nn.Sequential(
				nn.Conv2d(n_filters, 4, kernel_size=1, bias=False),
				nn.BatchNorm2d(4),
				nn.ReLU()
			)
			self.head_V_fc = nn.Sequential(
				nn.Linear(4 * 9 * 9 + 16, 64),
				nn.ReLU(),
				nn.Linear(64, self.num_players)
			)

		# else:
		# 	raise Exception(f'Warning, unknown NN version {self.version}')

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
		# input_data is (Batch, H, W, C) -> (Batch, 9, 9, 4)
		
		# 1. Separate Spatial data and Meta data
		# Spatial is channels 0, 1, 2. Meta is channel 3.
		spatial_data = input_data[:, :, :, :3].permute(0, 3, 1, 2) # -> (Batch, 3, 9, 9)
		
		# Metadata is extracted from specific cells defined in AbaloneLogicNumba.py
		# (0, 0) : P0 score, (0, 1) : P1 score, (0, 2) : Round
		meta_data = input_data[:, 0, 0:3, 3] # -> (Batch, 3)
		
		# 2. Process trunk
		x_features = self.first_layer(spatial_data)
		x_features = self.trunk(x_features)
		meta_embedding = self.meta_fc(meta_data)
		
		# 3. Policy computation
		pi_spatial = self.head_PI(x_features) # -> (Batch, 42, 9, 9)
		# CRITICAL: We must permute to match Numba encoding: r * 9 * 42 + q * 42 + plane
		pi_spatial = pi_spatial.permute(0, 2, 3, 1).contiguous() # -> (Batch, 9, 9, 42)
		pi = pi_spatial.flatten(1)                               # -> (Batch, 3402)
		pi = torch.where(valid_actions, pi, self.lowvalue)
		
		# 4. Value computation
		v_features = self.head_V_conv(x_features)
		v_features = torch.flatten(v_features, 1)
		v_combined = torch.cat([v_features, meta_embedding], dim=1)
		v = self.head_V_fc(v_combined)

		return F.log_softmax(pi, dim=1), torch.tanh(v)