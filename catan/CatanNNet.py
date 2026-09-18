import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
	from .CatanConstants import *
except ImportError:                                  # run directly from catan/
	from CatanConstants import *

###############################################################################
# ARCHITECTURE
#
# One message-passing trunk over the EXACT incidence graph of the board, plus a
# global token. 77 tokens for 3 players:
#     54 vertices | 19 hexes | P players | 1 global
# An EDGE has no token: its logit comes from a rank-16 bilinear form over its two
# endpoints, so every edge still gets its own logit -- never a mean over options.
#
# Versions (see --nn-version):
#     V10 "tiny"  d=32 L=4     V11 "base" d=48 L=3     V12 "wide" d=64 L=2
#
# EQUIVARIANCE is a hard requirement, not a nicety: get_symmetries() augments
# every sample 12x, and the augmented (state, policy) pairs are only consistent
# if f(sigma . s) == sigma . f(s). Three rules keep it true, all checked by
# CatanNNetTest.py:
#   1. the 3 edge slots of a vertex are PERMUTED by an isometry -> one shared
#      embedding table summed over the slots, never three separate tables;
#   2. neighbour aggregation is a mean -> permutation invariant, no positional
#      weights anywhere;
#   3. the edge head is symmetric in (u, v) -> elementwise product.
# Any static per-vertex feature must be isometry-INVARIANT. Degree and number of
# adjacent land hexes qualify (an isometry is a graph automorphism); a vertex
# index would not.
#
# HIDDEN INFORMATION: the search feeds a complete sampled world, so nothing
# changes here. The player token takes the public totals as explicit inputs and
# carries a `masked` flag, derived from the state (detail no longer summing to
# the total). Without that flag the network would have to learn to distinguish
# "holds 0 cards" from "hand hidden", which is exactly the kind of ambiguity that
# costs months. A standing trade offer is PUBLIC and is encoded on its author's
# own token, never masked.
###############################################################################

VERSIONS = {
	10: dict(dim=32, layers=4, edge_rank=16, name='tiny'),
	11: dict(dim=48, layers=3, edge_rank=16, name='base'),
	12: dict(dim=64, layers=2, edge_rank=16, name='wide'),
}

N_TOKENS = N_VERTICES + N_HEXES + N_PLAYERS + 1
TOK_HEX = N_VERTICES
TOK_PLAYER = N_VERTICES + N_HEXES
TOK_GLOBAL = TOK_PLAYER + N_PLAYERS
PAD = N_TOKENS                                       # index of the zero row
MAX_NBR = 6


def _idx(col, n):
	"""Round, cast to long, clamp into [0, n).

	export_and_load_onnx() (GenericNNetWrapper.py) traces the graph on a DUMMY
	board of unconstrained torch.randn floats, purely to capture shapes -- the
	values are never meant to be meaningful, and the same happens under
	FlopCountAnalysis and torch.jit tracing generally. Every embedding lookup
	fed from the board must therefore tolerate arbitrary input without raising:
	plain `.long()` indexing does not, since a random float can round outside
	[0, num_embeddings). Real boards are always integer-valued and in range,
	so the clamp is a no-op there -- this changes nothing for genuine positions,
	only makes tracing on garbage safe.
	"""
	return col.round().long().clamp(0, n - 1)

# Number of policy outputs produced by the global token: the blocks that are not
# anchored on a vertex, an edge or a hex.
N_GLOBAL_HEAD_A = 1 + 2                              # buy dev, play dev (knight, road building)
N_GLOBAL_HEAD_B = 1 + N_RESOURCES + 15 + 20 + N_RESOURCES + 1   # roll, mono, yop, bank trade, discard, end
# Player trade: 55 RECV + 55 GIVE + OK + NO + P accept = 112 + P. Same formula as
# when the block was still reserved -- it only works because the block stays LAST
# in the id space, hence one contiguous slice of the global token's output.
N_GLOBAL_HEAD_C = N_ACTIONS - N_ACTIONS_V1


def _build_neighbours():
	"""Padded neighbour table, one row per token, MAX_NBR columns.

	Order inside a row is irrelevant: the aggregation is a mean. PAD points at an
	appended zero row, so padding contributes nothing.
	"""
	nbr = np.full((N_TOKENS, MAX_NBR), PAD, dtype=np.int64)
	deg = np.zeros((N_TOKENS, 1), dtype=np.float32)
	for v in range(N_VERTICES):
		lst = [int(o) for o in VERTEX_TO_VERTEX[v] if o != NO_VERTEX]
		lst += [TOK_HEX + int(h) for h in VERTEX_TO_HEX[v] if h != NO_HEX]
		nbr[v, :len(lst)] = lst
		deg[v] = len(lst)
	for h in range(N_HEXES):
		lst = [int(v) for v in HEX_TO_VERTEX[h]]
		nbr[TOK_HEX + h, :len(lst)] = lst
		deg[TOK_HEX + h] = len(lst)
	for p in range(N_PLAYERS):
		nbr[TOK_PLAYER + p, 0] = TOK_GLOBAL
		deg[TOK_PLAYER + p] = 1
	lst = [TOK_PLAYER + p for p in range(N_PLAYERS)]
	nbr[TOK_GLOBAL, :len(lst)] = lst
	deg[TOK_GLOBAL] = len(lst)
	return nbr, deg


class CatanNNet(nn.Module):
	def __init__(self, game, args):
		super().__init__()
		# The project convention is (game, nn_args): nn_args is the FULL dict
		# built in main.py (lr, dropout, epochs, nn_version, ...), not just the
		# version number -- catan/NNet.py's init_nnet() calls nn_model(game,
		# nn_args) directly, like every other game. `int` is accepted too, for
		# CatanNNetTest.py's convenience and for ad-hoc construction.
		if isinstance(args, int):
			version = args
		elif isinstance(args, dict):
			version = args['nn_version']
		else:
			version = args.nn_version
		if version == -1:
			version = min(VERSIONS)
		elif version not in VERSIONS:
			raise ValueError(f'Catan supports nn-version {sorted(VERSIONS)}, got {version}')
		self.version = version
		cfg = VERSIONS[version]
		d, self.n_layers, rank = cfg['dim'], cfg['layers'], cfg['edge_rank']
		self.dim = d
		self.num_players = game.num_players if game is not None else N_PLAYERS
		P = self.num_players

		# ---- static topology, as buffers so they follow .to() and ONNX --------
		nbr, deg = _build_neighbours()
		self.register_buffer('nbr', torch.from_numpy(nbr), persistent=False)
		self.register_buffer('deg', torch.from_numpy(deg), persistent=False)
		self.register_buffer('edge_u', torch.from_numpy(EDGE_TO_VERTEX[:, 0].astype(np.int64)), persistent=False)
		self.register_buffer('edge_v', torch.from_numpy(EDGE_TO_VERTEX[:, 1].astype(np.int64)), persistent=False)
		v_deg = (VERTEX_TO_EDGE != NO_EDGE).sum(1).astype(np.int64)      # 2 or 3, isometry-invariant
		v_land = (VERTEX_TO_HEX != NO_HEX).sum(1).astype(np.int64)       # 1..3, isometry-invariant
		self.register_buffer('v_deg', torch.from_numpy(v_deg), persistent=False)
		self.register_buffer('v_land', torch.from_numpy(v_land), persistent=False)

		# ---- encoders ---------------------------------------------------------
		self.emb_building = nn.Embedding(3, d)
		self.emb_owner = nn.Embedding(P + 1, d)
		self.emb_port = nn.Embedding(N_PORT_TYPES, d)
		self.emb_edge = nn.Embedding(P + 1, d)        # SHARED across the 3 slots
		self.emb_vdeg = nn.Embedding(4, d)
		self.emb_vland = nn.Embedding(4, d)

		self.emb_hextype = nn.Embedding(N_HEX_TYPES, d)
		self.emb_pips = nn.Embedding(6, d)
		self.emb_token = nn.Embedding(N_TOKEN_IDS, d)
		self.emb_robber = nn.Embedding(2, d)

		# +2*N_RESOURCES: this player's standing trade offer (what it asks, what it
		# offers). It lives on the player's own row D, so it is encoded into the
		# player's OWN token rather than the global one -- the net is asked to
		# answer and to counter, and an offer whose author is ambiguous would make
		# both undecidable.
		self.n_player_feat = 3 * N_RESOURCES + 3 + 7 + 3 + 6 + 2 * N_RESOURCES
		self.player_proj = nn.Linear(self.n_player_feat, d)
		self.emb_seat = nn.Embedding(P, d)
		self.emb_masked = nn.Embedding(2, d)
		self.emb_trade_status = nn.Embedding(4, d)    # TRADE_NONE/COMPOSING/OFFERED/REFUSED

		self.n_global_feat = 2 * N_RESOURCES + N_DEV_TYPES + 4 + 1
		self.global_proj = nn.Linear(self.n_global_feat, d)
		self.emb_phase = nn.Embedding(N_PHASES, d)
		self.emb_dice = nn.Embedding(13, d)
		self.emb_turn = nn.Embedding(P, d)

		self.emb_kind = nn.Embedding(4, d)            # vertex / hex / player / global
		self.in_norm = nn.LayerNorm(d)

		# ---- trunk -------------------------------------------------------------
		self.msg = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(self.n_layers)])
		self.upd = nn.ModuleList([nn.Linear(3 * d, d) for _ in range(self.n_layers)])
		self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.n_layers)])

		# ---- heads -------------------------------------------------------------
		self.head_vertex = nn.Linear(d, 2)            # settlement, city
		self.head_hex = nn.Linear(d, P)               # robber destination x victim
		self.head_edge_proj = nn.Linear(d, rank, bias=False)
		self.head_edge_w = nn.Parameter(torch.randn(rank) * (rank ** -0.5))
		self.head_edge_b = nn.Parameter(torch.zeros(1))
		self.head_global = nn.Linear(d, N_GLOBAL_HEAD_A + N_GLOBAL_HEAD_B + N_GLOBAL_HEAD_C)

		self.value_score = nn.Linear(d, 1)            # attention pooling
		self.value_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, P))

	# ------------------------------------------------------------------ encode
	def _encode(self, board):
		B = board.shape[0]
		P = self.num_players
		verts = board[:, :N_VERTICES, :]
		hexes = board[:, TOK_HEX:TOK_HEX + N_HEXES, :]
		players = board[:, TOK_PLAYER:TOK_PLAYER + 4 * P, :].reshape(B, P, 4 * N_COLS)
		glob = board[:, TOK_PLAYER + 4 * P:, :].reshape(B, 2 * N_COLS)

		hv = (self.emb_building(_idx(verts[:, :, V_BUILDING], 3))
		      + self.emb_owner(_idx(verts[:, :, V_OWNER], P + 1))
		      + self.emb_port(_idx(verts[:, :, V_PORT], N_PORT_TYPES))
		      + self.emb_vdeg(self.v_deg).unsqueeze(0)
		      + self.emb_vland(self.v_land).unsqueeze(0))
		# one shared table, summed over the 3 slots: an isometry permutes them
		for k in range(3):
			hv = hv + self.emb_edge(_idx(verts[:, :, V_EDGE0 + k], P + 1))

		hh = (self.emb_hextype(_idx(hexes[:, :, H_TYPE], N_HEX_TYPES))
		      + self.emb_pips(_idx(hexes[:, :, H_PIPS], 6))
		      + self.emb_token(_idx(hexes[:, :, H_TOKEN], N_TOKEN_IDS))
		      + self.emb_robber(_idx(hexes[:, :, H_ROBBER], 2)))

		# --- player tokens: counts as floats, plus seat and masking flag -------
		res = players[:, :, PA_RESOURCES:PA_RESOURCES + N_RESOURCES] / float(BANK_PER_RESOURCE)
		devp = players[:, :, PA_DEV_PLAYABLE:PA_DEV_PLAYABLE + N_DEV_TYPES] / 5.
		devn = players[:, :, N_COLS + PB_DEV_NEW:N_COLS + PB_DEV_NEW + N_DEV_TYPES] / 5.
		tot = torch.stack([players[:, :, PA_TOTAL_RES] / float(BANK_PER_RESOURCE),
		                   players[:, :, PA_TOTAL_DEV] / 25.,
		                   players[:, :, 2 * N_COLS + PC_TOTAL_DEV_NEW] / 5.], dim=-1)
		misc = torch.stack([players[:, :, N_COLS + PB_KNIGHTS] / 14.,
		                    players[:, :, N_COLS + PB_SETTLEMENTS_LEFT] / float(MAX_SETTLEMENTS),
		                    players[:, :, N_COLS + PB_CITIES_LEFT] / float(MAX_CITIES),
		                    players[:, :, N_COLS + PB_ROADS_LEFT] / float(MAX_ROADS),
		                    players[:, :, N_COLS + PB_ROAD_LENGTH] / float(MAX_ROADS),
		                    players[:, :, N_COLS + PB_HAS_ROAD],
		                    players[:, :, N_COLS + PB_HAS_ARMY]], dim=-1)
		vp = torch.stack([players[:, :, 2 * N_COLS + PC_VP_PUBLIC] / float(VP_TO_WIN),
		                  players[:, :, 2 * N_COLS + PC_VP_DEV] / 5.,
		                  players[:, :, 2 * N_COLS + PC_DEV_PLAYED_THIS_TURN]], dim=-1)
		ports = players[:, :, 2 * N_COLS + PC_PORTS:2 * N_COLS + PC_PORTS + 6]
		# row D: the standing offer. Counts are 0..3, so /3 puts them on the same
		# scale as the other normalised counts feeding this projection.
		t_recv = players[:, :, 3 * N_COLS + PD_TRADE_RECV:3 * N_COLS + PD_TRADE_RECV + N_RESOURCES] / 3.
		t_give = players[:, :, 3 * N_COLS + PD_TRADE_GIVE:3 * N_COLS + PD_TRADE_GIVE + N_RESOURCES] / 3.
		pf = torch.cat([res, devp, devn, tot, misc, vp, ports, t_recv, t_give], dim=-1)

		# masked iff the detail no longer sums to the public total
		detail_r = players[:, :, PA_RESOURCES:PA_RESOURCES + N_RESOURCES].sum(-1)
		detail_d = (players[:, :, PA_DEV_PLAYABLE:PA_DEV_PLAYABLE + N_DEV_TYPES].sum(-1)
		            + players[:, :, N_COLS + PB_DEV_NEW:N_COLS + PB_DEV_NEW + N_DEV_TYPES].sum(-1))
		masked = (((detail_r - players[:, :, PA_TOTAL_RES]).abs()
		           + (detail_d - players[:, :, PA_TOTAL_DEV]).abs()) > 0.5).long()

		seat = torch.arange(P, device=board.device)
		hp = (self.player_proj(pf) + self.emb_seat(seat).unsqueeze(0) + self.emb_masked(masked)
		      + self.emb_trade_status(_idx(players[:, :, 3 * N_COLS + PD_TRADE_STATUS], 4)))

		# --- global token -------------------------------------------------------
		rnd = (glob[:, N_COLS + GB_ROUND_HI] * 100. + glob[:, N_COLS + GB_ROUND_LO]) / float(MAX_ROUNDS)
		gf = torch.cat([
			glob[:, GA_BANK:GA_BANK + N_RESOURCES] / float(BANK_PER_RESOURCE),
			glob[:, GA_DEV_DECK:GA_DEV_DECK + N_DEV_TYPES] / 14.,
			glob[:, N_COLS + GB_DEV_PLAYED:N_COLS + GB_DEV_PLAYED + N_DEV_TYPES] / 14.,
			torch.stack([rnd,
			             glob[:, N_COLS + GB_PENDING_COUNT] / 2.,
			             glob[:, N_COLS + GB_SETUP_STEP] / float(2 * P),
			             glob[:, N_COLS + GB_CHANCE_COUNTER] / 100.,
			             glob[:, N_COLS + GB_PLAYER_TRADE_DONE]], dim=-1),
		], dim=-1)
		hg = (self.global_proj(gf)
		      + self.emb_phase(_idx(glob[:, GA_PHASE], N_PHASES))
		      + self.emb_dice(_idx(glob[:, GA_DICE], 13))
		      + self.emb_turn(_idx(glob[:, N_COLS + GB_TURN_PLAYER], P))).unsqueeze(1)

		kind = torch.cat([
			torch.zeros(N_VERTICES, dtype=torch.long, device=board.device),
			torch.ones(N_HEXES, dtype=torch.long, device=board.device),
			torch.full((P,), 2, dtype=torch.long, device=board.device),
			torch.full((1,), 3, dtype=torch.long, device=board.device)])
		h = torch.cat([hv, hh, hp, hg], dim=1) + self.emb_kind(kind).unsqueeze(0)
		return self.in_norm(h)

	# ------------------------------------------------------------------- trunk
	def _trunk(self, h):
		B = h.shape[0]
		for i in range(self.n_layers):
			m = self.msg[i](h)
			m_pad = torch.cat([m, torch.zeros(B, 1, self.dim, dtype=m.dtype, device=m.device)], dim=1)
			# gather + mean: permutation invariant, and no MACs
			agg = m_pad[:, self.nbr.reshape(-1), :].reshape(B, N_TOKENS, MAX_NBR, self.dim).sum(2)
			agg = agg / self.deg
			g = m.mean(dim=1, keepdim=True).expand(-1, N_TOKENS, -1)
			h = self.norm[i](h + self.upd[i](torch.cat([m, agg, g], dim=-1)))
		return h

	# -------------------------------------------------------------------- heads
	def forward(self, board, valid_actions):
		h = self._trunk(self._encode(board))
		hv = h[:, :N_VERTICES, :]
		hh = h[:, TOK_HEX:TOK_HEX + N_HEXES, :]
		hg = h[:, TOK_GLOBAL, :]

		# edges: rank-16 bilinear form, symmetric in (u, v)
		pr = self.head_edge_proj(hv)
		e_logit = ((pr[:, self.edge_u, :] * pr[:, self.edge_v, :]) * self.head_edge_w).sum(-1) + self.head_edge_b

		v_logit = self.head_vertex(hv)                             # (B, 54, 2)
		h_logit = self.head_hex(hh).reshape(hh.shape[0], -1)       # (B, 19*P)
		g_logit = self.head_global(hg)
		ga, gb, gc = torch.split(g_logit, [N_GLOBAL_HEAD_A, N_GLOBAL_HEAD_B, N_GLOBAL_HEAD_C], dim=-1)

		# concatenation follows the A_* offsets exactly, so no scatter is needed
		pi = torch.cat([e_logit, v_logit[:, :, 0], v_logit[:, :, 1], ga, h_logit, gb, gc], dim=-1)
		pi = torch.where(valid_actions, pi, torch.full_like(pi, -1e9))

		w = torch.softmax(self.value_score(h).squeeze(-1), dim=-1)
		pooled = torch.einsum('bt,btd->bd', w, h)
		v = torch.tanh(self.value_mlp(pooled))
		return F.log_softmax(pi, dim=-1), v


# catan/NNet.py (one per game, alongside CatanGame.py) is expected to be:
#
#     from .CatanNNet import CatanNNet as nn_model
#     from ..GenericNNetWrapper import GenericNNetWrapper
#
#     class NNetWrapper(GenericNNetWrapper):
#         def init_nnet(self, game, nn_args):
#             self.nnet = nn_model(game, nn_args)
#
# CatanNNet itself stays framework-agnostic (no GenericNNetWrapper import here),
# which is also what keeps CatanNNetTest.py runnable standalone.
