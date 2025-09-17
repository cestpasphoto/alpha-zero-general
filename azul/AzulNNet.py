import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import _make_divisible


class LinearNormActivation(nn.Module):
    def __init__(self, in_size, out_size, activation_layer, depthwise=False, channels=None):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size, bias=False)
        self.norm = nn.BatchNorm1d(channels if depthwise else out_size)
        self.activation = activation_layer(inplace=True) if activation_layer is not None else nn.Identity()
        self.depthwise = depthwise

    def forward(self, data_input):
        if self.depthwise:
            result = self.linear(data_input)
        else:
            result = self.linear(data_input.transpose(-1, -2)).transpose(-1, -2)
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

    def _scale(self, data_input):
        scale = self.avgpool(data_input)
        scale = self.fc1(scale.transpose(-1, -2)).transpose(-1, -2)
        scale = self.activation(scale)
        scale = self.fc2(scale.transpose(-1, -2)).transpose(-1, -2)
        return self.scale_activation(scale)

    def forward(self, data_input):
        scale = self._scale(data_input)
        return scale * data_input


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

    def forward(self, data_input):
        result = self.expand(data_input)
        result = self.depthwise(result)
        result = self.se(result)
        result = self.project(result)

        if self.use_res_connect:
            result += data_input

        return result


class AzulNNet(nn.Module):
    def __init__(self, game, args):
        self.nb_vect, self.vect_dim = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.num_players = game.num_players
        self.args = args
        self.version = args['nn_version']

        super(AzulNNet, self).__init__()
        if self.version == 84:
            self.first_layer = LinearNormActivation(self.nb_vect, self.nb_vect, None)
            confs = []
            confs += [InvertedResidual1d(self.nb_vect, 5*self.nb_vect, self.nb_vect, 6, False, "RE")]
            self.trunk = nn.Sequential(*confs)

            head_PI = [
                InvertedResidual1d(self.nb_vect, 5*self.nb_vect, 2*self.nb_vect, 6, True, "HS", setype='avg'),
                nn.Flatten(1),
                nn.Linear(self.nb_vect*12, self.action_size),
                nn.ReLU(),
                nn.Linear(self.action_size, self.action_size),
            ]
            self.output_layers_PI = nn.Sequential(*head_PI)

            head_V = [
                InvertedResidual1d(self.nb_vect, 2*self.nb_vect, self.nb_vect, 6, True, "HS", setype='avg'),
                nn.Flatten(1),
                nn.Linear(self.nb_vect*6, self.num_players),
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
        if self.version in [84]:
            x = input_data.view(-1, self.nb_vect, self.vect_dim) # no transpose
            x = self.first_layer(x)
            x = F.dropout(self.trunk(x), p=self.args['dropout'], training=self.training)
            v = self.output_layers_V(x)
            pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)

        else:
            raise Exception(f'Unsupported NN version {self.version}')

        return F.log_softmax(pi, dim=1), torch.tanh(v)

