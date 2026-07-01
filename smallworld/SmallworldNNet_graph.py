import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Real static map structure (per current NUMBER_PLAYERS): adjacency + terrain.
from .SmallworldMaps import connexity_matrix, descr
from .SmallworldConstants import DECK_SIZE, MAX_REDEPLOY

# =============================================================================
# Smallworld -- 3 graph architectures sharing ONE backbone (versions 72/73/74).
#
# Shared backbone (constant across the 3 arms, so the cleanups don't confound):
#   * TypedStem        : one encoder per heterogeneous row type (column meanings
#                        per SmallworldLogicNumba.py header; bitfields read as
#                        UNSIGNED bytes -- int8 high-bit bytes are reinterpreted).
#   * Static map input : per-area learned positional embedding + terrain features
#                        (descr: terrain id + cavern/magic/mine/lost-tribe/edge).
#                        Neither adjacency nor terrain is in self.state, so the
#                        net was structurally blind to the map; we inject it here.
#   * Global token     : pooled peoples/deck/status/invisible_deck.
#   * Heads            : per-area local (5 planes), PER-deck-token Choose,
#                        global actions from the global token, deeper value head.
#
# The ONLY thing that varies across arms is the area-mixing operator:
#   v72  GCNMixer    : message passing on connexity_matrix         (hard topology)
#   v73  AttnMixer*  : self-attention + additive adjacency bias     (soft topology)
#   v74  AttnMixer   : plain self-attention, NO adjacency bias      (CONTROL)
# v74 still gets positional embeddings + terrain, so it CAN learn topology
# implicitly -- making the ablation fair.
#
# FLOPs (D=40): GCN ~1.5-2 MFLOPs (4 layers), attention ~2.5-3 MFLOPs (2 layers,
# over the ~23-39 AREA tokens only). Dominant term per attention layer ~ 12*A*D^2;
# per GCN layer ~ 2*A*D^2 + A^2*D. 4-player (A=39) attention is the costliest --
# if a FLOP counter says >3 MFLOPs, drop d_model to 36.
#
# !!! VALIDATE BEFORE A TRAINING SLOT !!!
#   (1) forward pass on a real state: no assert fires (row partition, pi length).
#   (2) layout test: True indices of valid_actions map to the expected logits.
#   (3) FLOP count vs your 2-3 MFLOP target; tune d_model / *_layers.
#   (4) value-vector player order matches your (canonical-form) training target.
# =============================================================================

PPL_OFFSET, PPL_CARD = 15, 31      # people type in [-15, 15] (negated if declined)
PWR_OFFSET, PWR_CARD = 20, 41      # power in [-20, 20]
PLR_OFFSET, PLR_CARD = 1, 6        # owner/slot id in [-1, 4]
N_TERRAIN = 6                      # FORESTT..WATER
GLOBAL_ACTIONS = MAX_REDEPLOY + 2  # RedeployN(8) + Decline(1) + End(1)


def _norm_adj(adj):
    a = adj.float()
    a = a + torch.eye(a.size(0), dtype=a.dtype)
    deg = a.sum(-1).clamp(min=1.0)
    dinv = deg.pow(-0.5)
    return dinv.unsqueeze(1) * a * dinv.unsqueeze(0)


class TypedStem(nn.Module):
    """One encoder per row type. See SmallworldLogicNumba.py board description."""
    def __init__(self, d):
        super().__init__()
        self.emb_ppl = nn.Embedding(PPL_CARD, d)
        self.emb_pwr = nn.Embedding(PWR_CARD, d)
        self.emb_plr = nn.Embedding(PLR_CARD, d)
        self.terr_num = nn.Linear(5, d)   # territory cols 0,3,4,5,6 : nb, defType, defPwr, defTot, points
        self.ppl_num = nn.Linear(4, d)    # peoples cols 0,3,4,6     : nb, capInfo, pwrInfo, points (3,4 mixed; v1)
        self.deck_num = nn.Linear(2, d)   # deck cols 0,6            : nb, VP_to_win
        self.rs_num = nn.Linear(5, d)     # round_status cols 0,3,4,5,6 : nbOnMap, NETWDT, phase, defTot, points
        self.gs_num = nn.Linear(3, d)     # game_status cols 3,4,6   : round, curPplId, score
        self.inv_bits = nn.Linear(40, d)  # invisible_deck cols 0..4 : people(0-1)+power(2-4) bitfields -> 40 bits
        self.inv_num = nn.Linear(2, d)    # invisible_deck cols 5,6  : dice usage, random usage
        self.register_buffer('pow2', torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.long))

    def _ppwo(self, rows, owner=True):
        e = self.emb_ppl(torch.clamp(rows[..., 1].long() + PPL_OFFSET, 0, PPL_CARD - 1)) \
            + self.emb_pwr(torch.clamp(rows[..., 2].long() + PWR_OFFSET, 0, PWR_CARD - 1))
        if owner:
            e = e + self.emb_plr(torch.clamp(rows[..., 7].long() + PLR_OFFSET, 0, PLR_CARD - 1))
        return e

    def _bits(self, cols):
        # int8 bytes are stored signed; reinterpret as unsigned (% 256) before unpacking.
        v = (cols.long() % 256).unsqueeze(-1)
        return ((v // self.pow2) % 2).float()

    def forward(self, terr, ppl, deck, rs, gs, inv):
        terr_t = self._ppwo(terr) + self.terr_num(terr[..., [0, 3, 4, 5, 6]].float() / 10.0)
        ppl_t = self._ppwo(ppl) + self.ppl_num(ppl[..., [0, 3, 4, 6]].float() / 10.0)
        deck_t = self._ppwo(deck, owner=False) + self.deck_num(deck[..., [0, 6]].float() / 10.0)
        rs_t = self.emb_plr(torch.clamp(rs[..., 7].long() + PLR_OFFSET, 0, PLR_CARD - 1)) \
            + self.rs_num(rs[..., [0, 3, 4, 5, 6]].float() / 10.0)
        gs_t = self.emb_plr(torch.clamp(gs[..., 7].long() + PLR_OFFSET, 0, PLR_CARD - 1)) \
            + self.gs_num(gs[..., [3, 4, 6]].float() / 10.0)
        inv_t = self.inv_bits(self._bits(inv[..., 0:5]).flatten(1)) + self.inv_num(inv[..., [5, 6]].float() / 10.0)
        return terr_t, ppl_t, deck_t, rs_t, gs_t, inv_t


class GCNMixer(nn.Module):
    """v72 -- hard topology. GCN message passing + read/write global token."""
    def __init__(self, d, n_layers, adj_norm):
        super().__init__()
        self.register_buffer('adj', adj_norm)
        self.s = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n_layers))
        self.n = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n_layers))
        self.g = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n_layers))
        self.ln = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.gu = nn.ModuleList(nn.Linear(2 * d, d) for _ in range(n_layers))

    def forward(self, area, glob):
        for s, n, g, ln, gu in zip(self.s, self.n, self.g, self.ln, self.gu):
            neigh = torch.einsum('ij,bjd->bid', self.adj, area)
            out = s(area) + n(neigh) + g(glob).unsqueeze(1)
            area = area + F.relu(ln(out))
            glob = F.relu(gu(torch.cat([glob, area.mean(1)], dim=-1)))
        return area, glob


class AttnMixer(nn.Module):
    """v73 (use_adj_bias=True, soft topology) / v74 (use_adj_bias=False, control).
    Self-attention over AREA tokens + read/write global token. With bias, a single
    learned scalar boosts attention to graph neighbours (additive, can be overridden)."""
    def __init__(self, d, n_layers, n_heads, adj=None, use_adj_bias=False):
        super().__init__()
        self.use_bias = use_adj_bias
        if use_adj_bias:
            self.register_buffer('adj', adj.float())   # (A,A) 0/1, zero diagonal
            self.alpha = nn.Parameter(torch.tensor(1.0))
        self.attn = nn.ModuleList(nn.MultiheadAttention(d, n_heads, batch_first=True) for _ in range(n_layers))
        self.ln1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.ln2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.ff = nn.ModuleList(nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d)) for _ in range(n_layers))
        self.g = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n_layers))
        self.gu = nn.ModuleList(nn.Linear(2 * d, d) for _ in range(n_layers))

    def forward(self, area, glob):
        mask = (self.alpha * self.adj) if self.use_bias else None  # (A,A) float added to attn logits
        x = area
        for attn, ln1, ln2, ff, g, gu in zip(self.attn, self.ln1, self.ln2, self.ff, self.g, self.gu):
            x = x + g(glob).unsqueeze(1)
            h = ln1(x)
            a, _ = attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a
            x = x + ff(ln2(x))
            glob = F.relu(gu(torch.cat([glob, x.mean(1)], dim=-1)))
        return x, glob


class SmallworldGraphNNet(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        self.nb_vect, self.vect_dim = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.num_players = game.num_players
        self.args = args
        self.version = args['nn_version']
        D = args.get('d_model', 64)
        self.aux_score = args.get('aux_score', False)

        # Area count and state row partition (mirror copy_state / valid_moves).
        self.A = (self.action_size - (MAX_REDEPLOY + DECK_SIZE + 2)) // 5
        P = self.num_players
        self.s_terr = (0, self.A)
        self.s_ppl = (self.A, self.A + 3 * P)
        self.s_deck = (self.A + 3 * P, self.A + 3 * P + DECK_SIZE)
        self.s_rs = (self.s_deck[1], self.s_deck[1] + P)
        self.s_gs = (self.s_rs[1], self.s_rs[1] + P)
        self.s_inv = self.s_gs[1]
        assert self.s_inv + 1 == self.nb_vect, f"partition {self.s_inv + 1} != nb_vect {self.nb_vect}"

        adj = torch.as_tensor(connexity_matrix)
        assert adj.shape == (self.A, self.A), f"adj {tuple(adj.shape)} != ({self.A},{self.A})"
        assert descr.shape[0] == self.A, f"descr rows {descr.shape[0]} != A {self.A}"

        self.stem = TypedStem(D)
        # static map features added to AREA tokens
        self.pos_emb = nn.Embedding(self.A, D)
        self.terr_emb = nn.Embedding(N_TERRAIN, D)
        self.terr_flag = nn.Linear(5, D)
        self.register_buffer('area_idx', torch.arange(self.A))
        self.register_buffer('terrain_id', torch.as_tensor(descr[:, 0]).long())
        self.register_buffer('terrain_flags', torch.as_tensor(descr[:, 1:6]).float())

        self.glob_init = nn.Linear(5 * D, D)

        if self.version == 72:
            self.mixer = GCNMixer(D, args.get('graph_layers', 4), _norm_adj(adj))
        elif self.version == 73:
            self.mixer = AttnMixer(D, args.get('attn_layers', 2), args.get('n_heads', 4), adj=adj, use_adj_bias=True)
        elif self.version == 74:
            self.mixer = AttnMixer(D, args.get('attn_layers', 2), args.get('n_heads', 4), use_adj_bias=False)
        else:
            raise Exception(f'SmallworldGraphNNet: unsupported version {self.version}')

        self.head_local = nn.Linear(D, 5)
        self.head_choose = nn.Linear(D, 1)
        self.head_global = nn.Linear(D, GLOBAL_ACTIONS)
        self.head_v = nn.Sequential(nn.Linear(D, D // 2), nn.ReLU(), nn.Linear(D // 2, P))
        self.head_score = nn.Linear(D, P)  # optional aux head (final score margin)

        self.register_buffer('lowvalue', torch.FloatTensor([-1e8]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_data, valid_actions):
        x = input_data.view(-1, self.nb_vect, self.vect_dim)
        terr = x[:, self.s_terr[0]:self.s_terr[1]]
        ppl = x[:, self.s_ppl[0]:self.s_ppl[1]]
        deck = x[:, self.s_deck[0]:self.s_deck[1]]
        rs = x[:, self.s_rs[0]:self.s_rs[1]]
        gs = x[:, self.s_gs[0]:self.s_gs[1]]
        inv = x[:, self.s_inv]

        terr_t, ppl_t, deck_t, rs_t, gs_t, inv_t = self.stem(terr, ppl, deck, rs, gs, inv)

        # inject static map structure into area tokens
        map_feat = self.pos_emb(self.area_idx) + self.terr_emb(self.terrain_id) + self.terr_flag(self.terrain_flags)
        area = terr_t + map_feat.unsqueeze(0)

        glob = self.glob_init(torch.cat([ppl_t.mean(1), deck_t.mean(1), rs_t.mean(1), gs_t.mean(1), inv_t], dim=-1))
        area, glob = self.mixer(area, glob)

        local = self.head_local(area)                  # (B,A,5)
        choose = self.head_choose(deck_t).squeeze(-1)  # (B,DECK)
        g = self.head_global(glob)                     # (B,10)

        pi = torch.cat([
            local[..., 0], local[..., 1], local[..., 2], local[..., 3],  # Abandon/Attack/SpecPpl/SpecPwr
            g[:, 0:MAX_REDEPLOY],                                        # RedeployN (incl. skip = redeploy 0)
            local[..., 4],                                              # Redeploy 1 per area
            choose,                                                     # Choose from deck
            g[:, MAX_REDEPLOY:MAX_REDEPLOY + 1],                        # Decline
            g[:, MAX_REDEPLOY + 1:MAX_REDEPLOY + 2],                    # End
        ], dim=1)
        assert pi.size(1) == self.action_size, f"pi {pi.size(1)} != action_size {self.action_size}"

        v = self.head_v(glob)
        pi = torch.where(valid_actions, pi, self.lowvalue)
        # If aux_score: also return torch.tanh(self.head_score(glob)); target = normalized
        # final score margin per player; loss = small_weight * MSE (keep weight ~0.1-0.3).
        return F.log_softmax(pi, dim=1), torch.tanh(v)
