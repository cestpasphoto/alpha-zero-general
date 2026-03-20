import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import InvertedResidualConfig, InvertedResidual

class CatInTheMiddle(nn.Module):
	def __init__(self, layers2D, layers1D):
		super().__init__()
		self.layers2D, self.layers1D = layers2D, layers1D

	def forward(self, input2D, input1D):
		x = self.layers2D(input2D)
		x = torch.cat([torch.flatten(x, 1), torch.flatten(input1D, 1)], dim=-1)
		x = self.layers1D(x)
		return x

class SimpleHead(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_features, is_value_head=False):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(bottleneck_channels)
        self.is_value_head = is_value_head
        
        flat_size = bottleneck_channels * 5 * 5
        
        if is_value_head:
            self.fc1 = nn.Linear(flat_size, 64)
            self.fc2 = nn.Linear(64, out_features)
        else:
            self.fc = nn.Linear(flat_size, out_features)

    def forward(self, x):
        x = F.relu(self.bn(self.conv1x1(x)))
        x = torch.flatten(x, 1)
        
        if self.is_value_head:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = self.fc(x)
        return x

class HeadWithMeta(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, meta_features, out_features, is_value_head=False):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(bottleneck_channels)
        self.is_value_head = is_value_head
        
        flat_size = bottleneck_channels * 5 * 5
        
        if is_value_head:
            self.fc1 = nn.Linear(flat_size + meta_features, 64)
            self.fc2 = nn.Linear(64, out_features)
        else:
            self.fc = nn.Linear(flat_size + meta_features, out_features)

    def forward(self, x_spatial, x_meta):
        x = F.relu(self.bn(self.conv1x1(x_spatial)))
        x = torch.flatten(x, 1)
        # Concatenate spatial features with gods/metadata features
        x = torch.cat([x, x_meta], dim=1)
        
        if self.is_value_head:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = self.fc(x)
        return x

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

class SantoriniNNet(nn.Module):
	def __init__(self, game, args):
		# game params
		self.board_size = game.getBoardSize()
		self.action_size = game.getActionSize()
		self.num_players = 2
		self.args = args
		self.version = args['nn_version']

		super(SantoriniNNet, self).__init__()

		if self.version == -1:
			pass # Special case when loading empty NN from pit.py
		elif self.version == 66:
			n_filters = 32
			n_filters_first = n_filters//2
			n_filters_begin = n_filters_first
			n_filters_end = n_filters
			n_exp_begin, n_exp_end = n_filters_begin*3, n_filters_end*3
			depth = 12 - 2

			def inverted_residual(input_ch, expanded_ch, out_ch, use_se, activation, kernel=3, stride=1, dilation=1, width_mult=1):
				return InvertedResidual(InvertedResidualConfig(input_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation, width_mult), nn.BatchNorm2d)
			self.first_layer = nn.Conv2d(  2, n_filters_first, 3, padding=1, bias=False)
			confs  = [inverted_residual(n_filters_first if i==0 else n_filters_begin, n_exp_begin, n_filters_begin, False, "RE") for i in range(depth//2)]
			confs += [inverted_residual(n_filters_begin if i==0 else n_filters_end  , n_exp_end  , n_filters_end  , True , "HS") for i in range(depth//2)]
			confs += [inverted_residual(n_filters_end                               , n_exp_end  , n_filters      , True , "HS")]
			self.trunk = nn.Sequential(*confs)

			head_depth = 6
			n_exp_head = n_filters * 3
			head_PI = [inverted_residual(n_filters, n_exp_head, n_filters, True, "HS",) for i in range(head_depth)] + [
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size)
			]
			self.output_layers_PI = nn.Sequential(*head_PI)

			head_V = [inverted_residual(n_filters, n_exp_head, n_filters, True, "HS",) for i in range(head_depth)] + [
				nn.Flatten(1),
				nn.Linear(n_filters *5*5, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players)
			]
			self.output_layers_V = nn.Sequential(*head_V)

		elif self.version == 67:
			n_filters = 32
			n_filters_first = n_filters//2
			n_filters_begin = n_filters_first
			n_filters_end = n_filters
			n_exp_begin, n_exp_end = n_filters_begin*3, n_filters_end*3
			depth = 12 - 2

			def inverted_residual(input_ch, expanded_ch, out_ch, use_se, activation, kernel=3, stride=1, dilation=1, width_mult=1):
				return InvertedResidual(InvertedResidualConfig(input_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation, width_mult), nn.BatchNorm2d)
			self.first_layer = nn.Conv2d(  2, n_filters_first, 3, padding=1, bias=False)
			confs  = [inverted_residual(n_filters_first if i==0 else n_filters_begin, n_exp_begin, n_filters_begin, False, "RE") for i in range(depth//2)]
			confs += [inverted_residual(n_filters_begin if i==0 else n_filters_end  , n_exp_end  , n_filters_end  , True , "HS") for i in range(depth//2)]
			confs += [inverted_residual(n_filters_end                               , n_exp_end  , n_filters      , True , "HS")]
			self.trunk = nn.Sequential(*confs)

			head_depth = 6
			n_exp_head = n_filters * 3
			head_PI_2D = [inverted_residual(n_filters, n_exp_head, n_filters, True, "HS",) for i in range(head_depth)]
			head_PI_1D = [
				nn.Linear((n_filters+1) *5*5, self.action_size),
				nn.ReLU(),
				nn.Linear(self.action_size, self.action_size)
			]
			self.output_layers_PI = CatInTheMiddle(nn.Sequential(*head_PI_2D), nn.Sequential(*head_PI_1D))

			head_V_2D = [inverted_residual(n_filters, n_exp_head, n_filters, True, "HS",) for i in range(head_depth)]
			head_V_1D = [
				nn.Linear((n_filters+1) *5*5, self.num_players),
				nn.ReLU(),
				nn.Linear(self.num_players, self.num_players)
			]
			self.output_layers_V = CatInTheMiddle(nn.Sequential(*head_V_2D), nn.Sequential(*head_V_1D))

		elif self.version == 78 or self.version == 88:
			n_filters = 64
			depth = 10
			meta_features = 32
			
			def inverted_residual(input_ch, expanded_ch, out_ch, use_se, activation, kernel=3, stride=1, dilation=1, width_mult=1):
				return InvertedResidual(InvertedResidualConfig(input_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation, width_mult), nn.BatchNorm2d)
			
			self.first_layer = nn.Conv2d(2, n_filters, 3, padding=1, bias=False)
			
			# Expansion factor is 3 (n_filters * 3 = 192 internal channels)
			confs = [inverted_residual(n_filters, n_filters*3, n_filters, False, "RE") for _ in range(depth)]
			self.trunk = nn.Sequential(*confs)
			
			if self.version == 78:
				self.meta_fc = nn.Sequential(
					nn.Flatten(1),
					nn.Linear(5 * 5, meta_features),
					nn.ReLU()
				)
				# Keeping bottlenecks small to avoid exploding the parameter count in the dense layers
				self.head_PI = HeadWithMeta(n_filters, bottleneck_channels=4, meta_features=meta_features, out_features=self.action_size, is_value_head=False)
				self.head_V  = HeadWithMeta(n_filters, bottleneck_channels=2, meta_features=meta_features, out_features=self.num_players, is_value_head=True)
			else:
				self.head_PI = SimpleHead(n_filters, bottleneck_channels=4, out_features=self.action_size, is_value_head=False)
				self.head_V  = SimpleHead(n_filters, bottleneck_channels=2, out_features=self.num_players, is_value_head=True)

		elif self.version == 79 or self.version == 89:
			n_filters = 64
			depth = 5
			meta_features = 32
			
			self.first_layer = nn.Sequential(
				nn.Conv2d(2, n_filters, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(n_filters),
				nn.ReLU()
			)
			
			self.trunk = nn.Sequential(*[SimpleResBlock(n_filters) for _ in range(depth)])
			
			if self.version == 79:
				self.meta_fc = nn.Sequential(
					nn.Flatten(1),
					nn.Linear(5 * 5, meta_features),
					nn.ReLU()
				)
				self.head_PI = HeadWithMeta(n_filters, bottleneck_channels=2, meta_features=meta_features, out_features=self.action_size, is_value_head=False)
				self.head_V = HeadWithMeta(n_filters, bottleneck_channels=1, meta_features=meta_features, out_features=self.num_players, is_value_head=True)
			else:
				self.head_PI = SimpleHead(n_filters, bottleneck_channels=2, out_features=self.action_size, is_value_head=False)
				self.head_V = SimpleHead(n_filters, bottleneck_channels=1, out_features=self.num_players, is_value_head=True)
		
		else:
			raise Exception(f'Warning, unknown NN version {self.version}')

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
		# input_data is (N, H, W, C) typically (N, 5, 5, 3)
		if self.version == 1:
			x = input_data.transpose(-1, -2)
			x = torch.flatten(x, start_dim=1).unsqueeze(1)
			x = F.dropout(self.dense1d_1(x)     , p=self.args['dropout'], training=self.training)
			x = F.dropout(self.partialgpool_1(x), p=self.args['dropout'], training=self.training)
			x = F.dropout(self.dense1d_2(x)     , p=self.args['dropout'], training=self.training)
			x = F.dropout(self.partialgpool_2(x), p=self.args['dropout'], training=self.training)
			x = F.dropout(self.dense1d_3(x)     , p=self.args['dropout'], training=self.training)
			
			v = self.output_layers_V(x).squeeze(1)
			pi = torch.where(valid_actions, self.output_layers_PI(x).squeeze(1), self.lowvalue)

		elif self.version in [66]:
			x = input_data.transpose(-1, -2).reshape(-1, 3, 5, 5) # BUG, see V78
			x, data = x.split([2,1], dim=1)
			x = self.first_layer(x)
			x = self.trunk(x)
			v = self.output_layers_V(x)
			pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)

		elif self.version in [67]:
			x = input_data.transpose(-1, -2).reshape(-1, 3, 5, 5) # BUG, see V88
			x, data = x.split([2,1], dim=1)
			x = self.first_layer(x)
			x = self.trunk(x)
			v = self.output_layers_V(x, data)
			pi = torch.where(valid_actions, self.output_layers_PI(x, data), self.lowvalue)

		elif self.version in [78, 79]: # Generated by Gemini - with gods
			x = input_data.permute(0, 3, 1, 2)         # (Batch, H, W, C) -> (Batch, C, H, W)
			x_spatial, x_data = x.split([2, 1], dim=1) # Split spatial data (workers, levels) and metadata (gods/rounds)
			x_features = self.first_layer(x_spatial) # Process trunk
			x_features = self.trunk(x_features)
			meta_embedding = self.meta_fc(x_data) # Process metadata
			v = self.head_V(x_features, meta_embedding)
			pi = torch.where(valid_actions, self.head_PI(x_features, meta_embedding), self.lowvalue)

		elif self.version in [88, 89]: # Generated by Gemini - w/o gods
			x = input_data.permute(0, 3, 1, 2)    # (Batch, H, W, C) -> (Batch, C, H, W)
			x_spatial, _ = x.split([2, 1], dim=1) # We only keep the 2 spatial channels (workers, levels)
			x_features = self.first_layer(x_spatial)
			x_features = self.trunk(x_features)
			v = self.head_V(x_features)
			pi = torch.where(valid_actions, self.head_PI(x_features), self.lowvalue)

		return F.log_softmax(pi, dim=1), torch.tanh(v)
