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


############################## V85 building blocks #############################

def _shift(w, k, dim):
    """Shift a (..., 5, 5) wall by k cells along dim (+k: towards higher index), zero filled. ONNX-safe."""
    if k > 0:
        pad = torch.zeros_like(w.narrow(dim, 0, k))
        return torch.cat([pad, w.narrow(dim, 0, 5 - k)], dim=dim)
    k = -k
    pad = torch.zeros_like(w.narrow(dim, 0, k))
    return torch.cat([w.narrow(dim, k, 5 - k), pad], dim=dim)


def _run_length(w, direction, dim):
    """Number of consecutive 1s next to each cell in one direction (excluding the cell)."""
    run, prod = torch.zeros_like(w), torch.ones_like(w)
    for k in range(1, 5):
        prod = prod * _shift(w, direction * k, dim)
        run = run + prod
    return run


class AzulFeatures(nn.Module):
    """Raw (9N+5, 6) state -> colour-aligned features. Equality / min / matmul only: ONNX-safe.

    In the raw state the column axis means a colour (bag, centre, factories), a
    pattern line (player_colours, player_row_numbers) or a wall position (walls).
    Everything is re-indexed here by (player, line, colour) so that downstream
    layers never have to decode it themselves.
    """
    def __init__(self, num_players):
        super().__init__()
        n = num_players
        self.n, self.f = n, 2 * n + 1
        # wall[i, (c+i)%5] -> wall_by_colour[i, c], as a constant 25x25 permutation
        m = torch.zeros(25, 25)
        for i in range(5):
            for c in range(5):
                m[i * 5 + (c + i) % 5, i * 5 + c] = 1.
        self.register_buffer('to_colour', m)
        self.register_buffer('colours', torch.arange(5, dtype=torch.float32))
        self.register_buffer('cap', torch.arange(1, 6, dtype=torch.float32))

    def forward(self, x):
        n, f = self.n, self.f
        o_col, o_num, o_wall = 4 + f, 4 + f + n, 4 + f + 2 * n
        r = {}
        r['scores'] = x[:, 0, :n]                                     # (B, n)
        r['round'] = x[:, 0, n:n + 1]                                 # (B, 1)
        r['bag'], r['disc'] = x[:, 1, :5], x[:, 2, :5]                # (B, 5)
        r['tok_centre'] = x[:, 3, 5:6]                                # (B, 1)
        r['src'] = x[:, 3:4 + f, :5]                                  # (B, F+1, 5), source 0 = centre
        pcol = x[:, o_col:o_col + n, :5]                              # (B, n, 5) colour id per line, -1 if empty
        r['ptok'] = x[:, o_col:o_col + n, 5]                          # (B, n)
        pnum = x[:, o_num:o_num + n, :5]                              # (B, n, 5)
        r['floor'] = x[:, o_num:o_num + n, 5]                         # (B, n)
        wall = x[:, o_wall:o_wall + 5 * n, :5].reshape(-1, n, 5, 5)   # (B, n, line, position)

        line_col = (pcol.unsqueeze(-1) == self.colours).float()       # (B, n, line, colour)
        empty = (pnum == 0).float()
        space = self.cap - pnum                                       # (B, n, line)
        wall_col = (wall.reshape(-1, n, 25) @ self.to_colour).reshape(-1, n, 5, 5)
        ok = empty.unsqueeze(-1) * (1. - wall_col) + line_col * (space > 0).float().unsqueeze(-1)

        # Points scored if a tile is placed on each empty wall cell (official adjacency rule)
        h = _run_length(wall, 1, 3) + _run_length(wall, -1, 3)
        v = _run_length(wall, 1, 2) + _run_length(wall, -1, 2)
        hp, vp = (h > 0).float(), (v > 0).float()
        place = (hp * (h + 1) + vp * (v + 1) + (1. - hp) * (1. - vp)) * (1. - wall)
        place_col = (place.reshape(-1, n, 25) @ self.to_colour).reshape(-1, n, 5, 5)

        r.update(pnum=pnum, space=space, line_col=line_col, wall=wall, wall_col=wall_col, ok=ok, place_col=place_col)
        return r


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


N_LINE_FEAT = 8 + 6   # see _line_features, plus line one-hot
N_VALUE_FEAT = 7      # see _value_features (V86)


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
                nn.LeakyReLU(0.01),  # leaky: a num_players-wide ReLU can die and freeze the value head
                nn.Linear(self.num_players, self.num_players),
            ]
            self.output_layers_V = nn.Sequential(*head_V)

        elif self.version in (85, 86):
            # Colour-aligned tokens + factory-equivariant policy head:
            #   tokens = 5 colours, centre + F factories, N seats, 1 global context
            #   factories share one role embedding and no position, so the net is
            #   exactly equivariant to factory permutations (augmentation still valid)
            #   logit(source, colour, line) = shared MLP(source token, colour token,
            #   exact consequences of the move for me: tiles to line/floor, line completion, wall points)
            n, f = self.num_players, 2 * self.num_players + 1
            D = args.get('v85_dim', 64)
            self.feat = AzulFeatures(n)
            self.tok_colour = nn.Linear(10 + 26 * n, D)
            self.tok_source = nn.Linear(9, D)
            self.tok_seat = nn.Linear(76, D)
            self.tok_glob = nn.Linear(5, D)
            self.role_colour = nn.Parameter(0.02 * torch.randn(5, D))
            self.role_centre = nn.Parameter(0.02 * torch.randn(1, D))
            self.role_factory = nn.Parameter(0.02 * torch.randn(1, D))
            self.role_seat = nn.Parameter(0.02 * torch.randn(n, D))
            self.role_glob = nn.Parameter(0.02 * torch.randn(1, D))
            self.blocks = nn.ModuleList([TokenBlock(D, args.get('v85_heads', 4)) for _ in range(args.get('v85_blocks', 3))])
            self.norm_out = nn.LayerNorm(D)

            self.pol_src = nn.Linear(D, D)
            self.pol_col = nn.Linear(D, D, bias=False)
            # Linear(concat(h, line features)) split in two so that the h part is
            # computed once per (source, colour) instead of once per line
            self.pol_hid_h = nn.Linear(D, D)
            self.pol_hid_l = nn.Linear(N_LINE_FEAT, D, bias=False)
            self.pol_final = nn.Linear(D, 1)
            if self.version == 85:
                self.value = nn.Sequential(nn.Linear(3 * D, D), nn.ReLU(), nn.Linear(D, 1))
            else:
                # V86: the V85 value head over-fits (cross-fit on a fixed buffer: held-out
                # MSE 0.88 -> 0.98 while the policy keeps improving). Value now reads the
                # trunk through a stop-gradient (the trunk is shaped by the policy only),
                # sees the other seats and a few exact relative features of the score.
                self.value = nn.Sequential(nn.Linear(4 * D + N_VALUE_FEAT, D), nn.ReLU(), nn.Linear(D, 1))

            is_centre = torch.zeros(f + 1, 1); is_centre[0] = 1.
            self.register_buffer('is_centre', is_centre)
            self.register_buffer('line_onehot', torch.eye(6))
            self.register_buffer('hist_k', torch.arange(1, 5, dtype=torch.float32))

        else:
            raise Exception(f'Unsupported NN version {self.version}')

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
        # input_data is (N, H, C) typically (N, 23, 6)
        if self.version in [84]:
            x = input_data.view(-1, self.nb_vect, self.vect_dim) # no transpose
            x = self.first_layer(x)
            x = F.dropout(self.trunk(x), p=self.args['dropout'], training=self.training)
            v = self.output_layers_V(x)
            pi = torch.where(valid_actions, self.output_layers_PI(x), self.lowvalue)

        elif self.version in (85, 86):
            x = input_data.view(-1, self.nb_vect, self.vect_dim)
            r = self.feat(x)
            tok = self._tokens(r)
            for blk in self.blocks:
                tok = blk(tok)
            tok = F.dropout(self.norm_out(tok), p=self.args['dropout'], training=self.training)
            n, s = self.num_players, 2 * self.num_players + 2
            t_col, t_src, t_seat, t_glob = tok.split([5, s, n, 1], dim=1)

            # Policy: (source, colour) pairs, then one logit per destination line
            h = F.relu(self.pol_src(t_src).unsqueeze(2) + self.pol_col(t_col).unsqueeze(1))       # (B, S, 5, D)
            lf = self._line_features(r)                                                            # (B, S, 5, 6, 14)
            z = F.relu(self.pol_hid_h(h).unsqueeze(3) + self.pol_hid_l(lf))                          # (B, S, 5, 6, D)
            logits = self.pol_final(z).squeeze(-1).flatten(1)                                      # (B, S*30)
            pi = torch.where(valid_actions, logits, self.lowvalue)

            # Value: one head shared by all seats (seat 0 = player to move)
            if self.version == 85:
                pooled = tok.mean(1, keepdim=True)
                v_in = torch.cat([t_seat, t_glob.expand(-1, n, -1), pooled.expand(-1, n, -1)], dim=-1)
            else:
                t_seat, t_glob, pooled = t_seat.detach(), t_glob.detach(), tok.detach().mean(1, keepdim=True)
                others = (t_seat.sum(1, keepdim=True) - t_seat) / (n - 1)
                v_in = torch.cat([t_seat, others, t_glob.expand(-1, n, -1), pooled.expand(-1, n, -1), self._value_features(r)], dim=-1)
            v = self.value(v_in).squeeze(-1)                                                       # (B, n)

        else:
            raise Exception(f'Unsupported NN version {self.version}')

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def _value_features(self, r):
        # Per seat (B, n, N_VALUE_FEAT): score, gap to the best other seat, floor, complete rows / columns / colours, round
        n = self.num_players
        sc = r['scores']
        best_other = torch.stack([torch.cat([sc[:, :p], sc[:, p + 1:]], 1).amax(1) for p in range(n)], 1)
        wall = r['wall']
        rows, cols, colours = wall.sum(-1), wall.sum(-2), r['wall_col'].sum(2)
        return torch.stack([sc / 50., (sc - best_other) / 50., r['floor'] / 7.,
                            (rows == 5).float().sum(-1), (cols == 5).float().sum(-1), (colours == 5).float().sum(-1),
                            r['round'].expand(-1, n) / 6.], dim=-1)

    def _tokens(self, r):
        n, f = self.num_players, 2 * self.num_players + 1
        src, fac = r['src'], r['src'][:, 1:]
        B = src.shape[0]

        # Colour tokens (B, 5, 10 + 26n)
        hist = (fac.unsqueeze(-1) == self.hist_k).float().sum(1) / f                             # (B, 5, 4)
        shared = torch.cat([
            (r['bag'] / 20.).unsqueeze(-1), (r['disc'] / 20.).unsqueeze(-1), (src[:, 0] / 10.).unsqueeze(-1),
            (fac.sum(1) / 8.).unsqueeze(-1), (fac.amax(1) / 4.).unsqueeze(-1), ((fac > 0).float().sum(1) / f).unsqueeze(-1),
            hist], dim=-1)
        space = (r['space'] / 5.).unsqueeze(-1).expand(-1, -1, -1, 5)
        per_line = torch.stack([r['ok'], r['line_col'], space, r['wall_col'], r['place_col'] / 10.], dim=-1)  # (B, n, line, colour, 5)
        per_player = per_line.permute(0, 3, 1, 2, 4).reshape(B, 5, n * 25)
        colour_count = r['wall_col'].sum(2) / 5.                                                  # (B, n, colour)
        t_col = self.tok_colour(torch.cat([shared, per_player, colour_count.transpose(1, 2)], dim=-1)) + self.role_colour

        # Source tokens: centre then factories (B, F+1, 9)
        is_centre = self.is_centre.unsqueeze(0).expand(B, -1, -1)
        t_src = self.tok_source(torch.cat([
            src / 4., src.sum(-1, keepdim=True) / 4., (src > 0).float().sum(-1, keepdim=True) / 5.,
            is_centre, is_centre * r['tok_centre'].unsqueeze(1)], dim=-1))
        t_src = t_src + is_centre * self.role_centre + (1. - is_centre) * self.role_factory

        # Seat tokens (B, n, 76)
        wall = r['wall']
        rows, cols = wall.sum(-1), wall.sum(-2)
        t_seat = self.tok_seat(torch.cat([
            (r['scores'] / 50.).unsqueeze(-1), (r['floor'] / 7.).unsqueeze(-1), r['ptok'].unsqueeze(-1),
            wall.reshape(B, n, 25), r['pnum'] / 5., r['line_col'].reshape(B, n, 25),
            rows / 5., cols / 5., colour_count,
            (rows == 5).float().sum(-1, keepdim=True) / 5., (cols == 5).float().sum(-1, keepdim=True) / 5.,
            (colour_count == 1).float().sum(-1, keepdim=True) / 5.], dim=-1)) + self.role_seat

        # Global token (B, 1, 5)
        t_glob = self.tok_glob(torch.cat([
            r['round'] / 6., r['tok_centre'], r['bag'].sum(-1, keepdim=True) / 100.,
            r['disc'].sum(-1, keepdim=True) / 100., src.sum((1, 2)).unsqueeze(-1) / (4. * f)], dim=-1)).unsqueeze(1) + self.role_glob

        return torch.cat([t_col, t_src, t_seat, t_glob], dim=1)

    def _line_features(self, r):
        # Exact consequences of action (source s, colour c, line l) for the player to move
        cnt = r['src'].unsqueeze(-1)                                    # (B, S, colour, 1)
        compat = r['ok'][:, 0].transpose(1, 2).unsqueeze(1)             # (B, 1, colour, line)
        space = r['space'][:, 0].unsqueeze(1).unsqueeze(1)              # (B, 1, 1, line)
        place = r['place_col'][:, 0].transpose(1, 2).unsqueeze(1)       # (B, 1, colour, line)
        to_line = compat * torch.minimum(space, cnt)                    # (B, S, colour, line)
        completes = compat * (to_line == space).float()
        gain = completes * place / 10.
        zeros = torch.zeros_like(cnt)
        compat = torch.cat([compat.expand_as(to_line), torch.ones_like(cnt)], dim=-1)    # floor = line 5
        to_line = torch.cat([to_line, zeros], dim=-1)
        completes = torch.cat([completes, zeros], dim=-1)
        gain = torch.cat([gain, zeros], dim=-1)
        to_floor = cnt - to_line                                        # (B, S, colour, 6)
        take_tok = (self.is_centre.unsqueeze(0) * r['tok_centre'].unsqueeze(1)).unsqueeze(-1)  # (B, S, 1, 1)
        floor0 = r['floor'][:, 0].reshape(-1, 1, 1, 1)
        floor_after = torch.clamp(floor0 + to_floor + take_tok, max=7.) / 7.
        feats = torch.stack([
            compat, to_line / 5., to_floor / 5., completes, gain,
            take_tok.expand_as(to_floor), floor_after, (cnt / 4.).expand_as(to_floor)], dim=-1)  # (B, S, colour, 6, 8)
        onehot = self.line_onehot.reshape(1, 1, 1, 6, 6).expand(feats.shape[0], feats.shape[1], 5, -1, -1)
        return torch.cat([feats, onehot], dim=-1)

