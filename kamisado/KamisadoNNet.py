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

class ConvNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, activation_layer=None, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = activation_layer(inplace=True) if activation_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class InvertedResidual1d(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, kernel, use_hs, use_se, setype='avg'):
        super().__init__()

        self.use_res_connect = (in_channels == out_channels)

        activation_layer = nn.Hardswish if use_hs else nn.ReLU

        # expand (1x1 conv)
        if exp_channels != in_channels:
            self.expand = ConvNormActivation(in_channels, exp_channels, activation_layer=activation_layer, kernel_size=1)
        else:
            self.expand = nn.Identity()

        # depthwise conv (kernel x 1)
        self.depthwise = nn.Sequential(
            nn.Conv1d(exp_channels, exp_channels, kernel_size=kernel, padding=kernel//2, groups=exp_channels, bias=False),
            nn.BatchNorm1d(exp_channels),
            activation_layer(inplace=True)
        )

        # squeeze and excitation or identity
        if use_se:
            squeeze_channels = _make_divisible(exp_channels // 4, 8)
            self.se = SqueezeExcitation1d(exp_channels, squeeze_channels, scale_activation=nn.Hardsigmoid, setype=setype)
        else:
            self.se = nn.Identity()

        # project (1x1 conv, no activation)
        self.project = ConvNormActivation(exp_channels, out_channels, activation_layer=None, kernel_size=1)

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)

        if self.use_res_connect:
            out += x

        return out


class KamisadoNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.nb_vect, self.vect_dim = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.num_players = game.num_players
        self.args = args
        self.version = args['nn_version']

        super(KamisadoNNet, self).__init__()
        if self.version == 100:
            n_filters = 16
            exp_factor = 5
            self.first_layer = LinearNormActivation(self.nb_vect, n_filters, None)
            confs  = []
            confs += [InvertedResidual1d(n_filters, n_filters*exp_factor, n_filters, 5, False, "RE")]
            self.trunk = nn.Sequential(*confs)

            head_PI = [
                InvertedResidual1d(n_filters, n_filters*exp_factor, n_filters, 5, True, "HS", setype='max'),
                nn.Flatten(1),
                nn.Linear(n_filters * self.vect_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.action_size),
            ]
            self.output_layers_PI = nn.Sequential(*head_PI)

            head_V = [
                InvertedResidual1d(n_filters, n_filters*exp_factor, n_filters, 5, True, "HS", setype='max'),
                nn.Flatten(1),
                nn.Linear(n_filters * self.vect_dim, 32),
                nn.ReLU(),
                nn.Linear(32, self.num_players),
            ]
            self.output_layers_V = nn.Sequential(*head_V)
        elif self.version == 101:
            n_filters = 8
            exp_factor = 3
            self.first_layer = LinearNormActivation(self.nb_vect, n_filters, None)
            confs  = []
            confs += [InvertedResidual1d(n_filters, n_filters*exp_factor, n_filters, 3, False, "RE")]
            self.trunk = nn.Sequential(*confs)

            head_PI = [
                InvertedResidual1d(n_filters, n_filters*exp_factor, n_filters, 3, True, "HS", setype='max'),
                nn.Flatten(1),
                nn.Linear(n_filters * self.vect_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.action_size),
            ]
            self.output_layers_PI = nn.Sequential(*head_PI)

            head_V = [
                InvertedResidual1d(n_filters, n_filters*exp_factor, n_filters, 3, True, "HS", setype='max'),
                nn.Flatten(1),
                nn.Linear(n_filters * self.vect_dim, 16),
                nn.ReLU(),
                nn.Linear(16, self.num_players),
            ]
            self.output_layers_V = nn.Sequential(*head_V)
        elif self.version == 102:
            n_filters = 32
            exp_factor = 6
            self.first_layer = LinearNormActivation(59, n_filters, None)
            confs  = []
            confs += [InvertedResidual1d(n_filters, n_filters*exp_factor, n_filters, 5, False, "RE")]
            confs += [InvertedResidual1d(n_filters, n_filters*exp_factor, n_filters, 5, False, "RE")]
            self.trunk = nn.Sequential(*confs)

            head_PI = [
                InvertedResidual1d(n_filters, n_filters*exp_factor, n_filters, 3, True, "HS", setype='max'),
                nn.Flatten(1),
                nn.Linear(n_filters * self.vect_dim, self.action_size),
                nn.ReLU(),
                nn.Linear(self.action_size, self.action_size),
            ]
            self.output_layers_PI = nn.Sequential(*head_PI)

            head_V = [
                InvertedResidual1d(n_filters, n_filters*exp_factor, n_filters, 3, True, "HS", setype='max'),
                nn.Flatten(1),
                nn.Linear(n_filters * self.vect_dim, 32),
                nn.ReLU(),
                nn.Linear(32, self.num_players),
            ]
            self.output_layers_V = nn.Sequential(*head_V)
        elif self.version == 103:
            n_filters = 24
            exp_factor = 4  # moderate expansion factor

            # First layer: project from nb_vect (e.g., 59) to n_filters (24)
            self.first_layer = LinearNormActivation(self.nb_vect, n_filters, activation_layer=nn.ReLU)

            # Trunk: two depthwise separable blocks with/without SE
            self.trunk = nn.Sequential(
                InvertedResidual1d(n_filters, n_filters * exp_factor, n_filters, kernel=3, use_hs=True, use_se=True, setype='avg'),
                InvertedResidual1d(n_filters, n_filters * exp_factor, n_filters, kernel=3, use_hs=True, use_se=False),
            )

            flattened_size = n_filters * self.vect_dim  # typically 24 * 17 = 408

            # Policy head
            self.output_layers_PI = nn.Sequential(
                InvertedResidual1d(n_filters, n_filters * 3, n_filters, kernel=3, use_hs=True, use_se=False),
                nn.Flatten(1),
                nn.Linear(flattened_size, 256),
                nn.ReLU(),
                nn.Linear(256, self.action_size),  # self.action_size = 681
            )

            # Value head
            self.output_layers_V = nn.Sequential(
                InvertedResidual1d(n_filters, n_filters * 3, n_filters, kernel=3, use_hs=True, use_se=True, setype='max'),
                nn.Flatten(1),
                nn.Linear(flattened_size, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_players),  # self.num_players = 2
            )
        elif self.version == 104:
            n_filters = 12          # ↓ from 24
            exp_factor = 2          # ↓ from 4

            # First layer: linear projection
            self.first_layer = LinearNormActivation(self.nb_vect, n_filters, activation_layer=nn.ReLU)

            # Trunk: single depthwise block (removes 1 layer)
            self.trunk = nn.Sequential(
                InvertedResidual1d(n_filters, n_filters * exp_factor, n_filters, kernel=3, use_hs=True, use_se=False),
            )

            flattened_size = n_filters * self.vect_dim  # e.g., 12 * 17 = 204

            # Policy head: smaller and simpler
            self.output_layers_PI = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(flattened_size, 64),
                nn.ReLU(),
                nn.Linear(64, self.action_size),  # e.g., 681
            )

            # Value head: same pattern, shallower
            self.output_layers_V = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(flattened_size, 32),
                nn.ReLU(),
                nn.Linear(32, self.num_players),  # typically 2
            )
        elif self.version == 105:
            exp_factor = 3  # expansion factor smaller than 159

            # First layer: LinearNormActivation replacing 1x1 conv
            self.first_layer = LinearNormActivation(self.nb_vect, self.nb_vect, None)

            # Trunk with 1 InvertedResidual1d block with smaller channels
            self.trunk = nn.Sequential(
                InvertedResidual1d(
                    in_channels=self.nb_vect,
                    exp_channels=self.nb_vect * exp_factor,
                    out_channels=self.nb_vect,
                    kernel=3,
                    use_hs=True,
                    use_se=False
                )
            )

            n_filters = self.nb_vect
            # Policy head: simpler, smaller layers
            self.output_layers_PI = nn.Sequential(
                InvertedResidual1d(
                    in_channels=self.nb_vect,
                    exp_channels=self.nb_vect * exp_factor,
                    out_channels=self.nb_vect,
                    kernel=3,
                    use_hs=True,
                    use_se=False
                ),
                nn.Flatten(1),
                nn.Linear(n_filters * self.vect_dim, 128),  # assuming length=17
                nn.ReLU(),
                nn.Linear(128, self.action_size),
            )

            # Value head: similar simplification
            self.output_layers_V = nn.Sequential(
                InvertedResidual1d(
                    in_channels=self.nb_vect,
                    exp_channels=self.nb_vect * exp_factor,
                    out_channels=self.nb_vect,
                    kernel=3,
                    use_hs=True,
                    use_se=False
                ),
                nn.Flatten(1),
                nn.Linear(n_filters * self.vect_dim, 32),
                nn.ReLU(),
                nn.Linear(32, self.num_players),
            )
        elif self.version == 110:
            exp_factor = 2  # smaller expansion for speed

            # First layer stays minimal
            self.first_layer = LinearNormActivation(self.nb_vect, self.nb_vect, None)

            # Trunk: kernel size 1 for max speed
            self.trunk = nn.Sequential(
                InvertedResidual1d(
                    in_channels=self.nb_vect,
                    exp_channels=self.nb_vect * exp_factor,
                    out_channels=self.nb_vect,
                    kernel=1,
                    use_hs=False,
                    use_se=False,
                )
            )

            n_filters = self.nb_vect

            # Policy head: flatter, fewer operations
            self.output_layers_PI = nn.Sequential(
                InvertedResidual1d(
                    in_channels=n_filters,
                    exp_channels=n_filters * exp_factor,
                    out_channels=n_filters,
                    kernel=1,
                    use_hs=False,
                    use_se=False
                ),
                nn.Flatten(1),
                nn.Linear(n_filters * self.vect_dim, self.action_size),
            )

            # Value head: similar reduction
            self.output_layers_V = nn.Sequential(
                InvertedResidual1d(
                    in_channels=n_filters,
                    exp_channels=n_filters * exp_factor,
                    out_channels=n_filters,
                    kernel=1,
                    use_hs=False,
                    use_se=False
                ),
                nn.Flatten(1),
                nn.Linear(n_filters * self.vect_dim, self.num_players),
            )

        elif self.version == 111:
            exp_factor = 4  # still efficient
            trunk_blocks = []

            # First layer
            self.first_layer = LinearNormActivation(self.nb_vect, self.nb_vect, None)

            # Trunk: 3 residual blocks with increasing complexity
            trunk_blocks.append(InvertedResidual1d(
                in_channels=self.nb_vect,
                exp_channels=self.nb_vect * exp_factor,
                out_channels=self.nb_vect,
                kernel=1,
                use_hs=False,
                use_se=False,
            ))

            trunk_blocks.append(InvertedResidual1d(
                in_channels=self.nb_vect,
                exp_channels=self.nb_vect * exp_factor,
                out_channels=self.nb_vect,
                kernel=3,
                use_hs=True,
                use_se=False,
            ))

            trunk_blocks.append(InvertedResidual1d(
                in_channels=self.nb_vect,
                exp_channels=self.nb_vect * exp_factor,
                out_channels=self.nb_vect,
                kernel=3,
                use_hs=True,
                use_se=True,  # helpful for value accuracy
            ))

            self.trunk = nn.Sequential(*trunk_blocks)
            n_filters = self.nb_vect

            # Policy head
            self.output_layers_PI = nn.Sequential(
                InvertedResidual1d(
                    in_channels=n_filters,
                    exp_channels=n_filters * exp_factor,
                    out_channels=n_filters,
                    kernel=1,
                    use_hs=True,
                    use_se=False
                ),
                nn.Flatten(1),
                nn.Linear(n_filters * self.vect_dim, self.action_size),
            )

            # Value head
            self.output_layers_V = nn.Sequential(
                InvertedResidual1d(
                    in_channels=n_filters,
                    exp_channels=n_filters * exp_factor,
                    out_channels=n_filters,
                    kernel=3,
                    use_hs=True,
                    use_se=True
                ),
                nn.Flatten(1),
                nn.Linear(n_filters * self.vect_dim, self.num_players),
            )
        elif self.version == 106:
            exp_factor = 6  # Increased expansion factor (was 3)

            # First layer: LinearNormActivation replacing 1x1 conv
            self.first_layer = LinearNormActivation(self.nb_vect, self.nb_vect, None)

            # Trunk with 2 InvertedResidual1d blocks for more depth and width
            self.trunk = nn.Sequential(
                InvertedResidual1d(
                    in_channels=self.nb_vect,
                    exp_channels=self.nb_vect * exp_factor,
                    out_channels=self.nb_vect,
                    kernel=3,
                    use_hs=True,
                    use_se=False
                ),
                InvertedResidual1d(
                    in_channels=self.nb_vect,
                    exp_channels=self.nb_vect * exp_factor,
                    out_channels=self.nb_vect,
                    kernel=3,
                    use_hs=True,
                    use_se=False
                )
            )

            n_filters = self.nb_vect
            # Policy head: wider fully connected layers
            self.output_layers_PI = nn.Sequential(
                InvertedResidual1d(
                    in_channels=self.nb_vect,
                    exp_channels=self.nb_vect * exp_factor,
                    out_channels=self.nb_vect,
                    kernel=3,
                    use_hs=True,
                    use_se=False
                ),
                nn.Flatten(1),
                nn.Linear(n_filters * self.vect_dim, 256),  # Increased from 128
                nn.ReLU(),
                nn.Linear(256, self.action_size),
            )

            # Value head: wider fully connected layers
            self.output_layers_V = nn.Sequential(
                InvertedResidual1d(
                    in_channels=self.nb_vect,
                    exp_channels=self.nb_vect * exp_factor,
                    out_channels=self.nb_vect,
                    kernel=3,
                    use_hs=True,
                    use_se=False
                ),
                nn.Flatten(1),
                nn.Linear(n_filters * self.vect_dim, 64),  # Increased from 32
                nn.ReLU(),
                nn.Linear(64, self.num_players),
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
        if self.version in [100, 101, 102, 103, 104, 110]:
            x = input_data.view(-1, self.nb_vect, self.vect_dim) # no transpose
            x = self.first_layer(x)
            x = F.dropout(self.trunk(x), p=self.args['dropout'], training=self.training)
            v = self.output_layers_V(x)
            pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)
        elif self.version in [105, 106]:
            x = input_data.view(-1, self.nb_vect, self.vect_dim) # no transpose
            x = self.first_layer(x)
            x = self.trunk(x)
            v = self.output_layers_V(x)
            pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)
        else:
            raise Exception(f'Unsupported NN version {self.version}')

        return F.log_softmax(pi, dim=1), torch.tanh(v)

