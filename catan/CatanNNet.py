import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
	from .CatanConstants import *
except ImportError:
	from CatanConstants import *

# Token graph: 54 vertices + 19 hexes + P players + 1 global, one token each.
#   - each token type is encoded by a sum of embeddings of its categorical
#     fields (one stacked table and one gather, CatEmbed) plus a projection of
#     its numerical fields; batch-constant terms (kind, seat, vertex degree)
#     live in one static table;
#   - trunk layer: one Linear(d, 3d) split into self / neighbour / context
#     messages, GELU, residual, LayerNorm. Neighbours are aggregated with a dense
#     row-normalised adjacency matmul; the context is mean(all) + global token +
#     viewer token;
#   - heads: bilinear edge logits on the two endpoint vertices, vertex logits
#     for settlements / cities, hex x victim logits for the robber, the rest
#     from the global token; value from an attention pooling over all tokens.
# The FAST family (V13) encodes each token type with one one-hot + one matmul
# and reads the value from the global token + the token mean: latency at this
# size is set by the number of kernels, not by MACs.

VERSIONS = {
	10: dict(dim=32, layers=4, edge_rank=16, name='tiny'),
	11: dict(dim=48, layers=3, edge_rank=16, name='base'),
	12: dict(dim=64, layers=2, edge_rank=16, name='wide'),
	13: dict(dim=32, layers=2, edge_rank=16, name='fast', fast=True),
}

N_TOKENS = N_VERTICES + N_HEXES + N_PLAYERS + 1
TOK_HEX = N_VERTICES
TOK_PLAYER = N_VERTICES + N_HEXES
TOK_GLOBAL = TOK_PLAYER + N_PLAYERS
MAX_NBR = 6

N_GLOBAL_HEAD_A = 1 + 2
N_GLOBAL_HEAD_B = 1 + N_RESOURCES + 15 + 20 + N_RESOURCES + 1
N_GLOBAL_HEAD_C = N_ACTIONS - N_ACTIONS_V1


def _build_neighbours():
	nbr = np.full((N_TOKENS, MAX_NBR), 0, dtype=np.int64)
	wts = np.zeros((N_TOKENS, MAX_NBR, 1), dtype=np.float32)
	def put(t, lst):
		nbr[t, :len(lst)] = lst
		wts[t, :len(lst), 0] = 1. / len(lst)
	for v in range(N_VERTICES):
		put(v, [int(o) for o in VERTEX_TO_VERTEX[v] if o != NO_VERTEX]
		     + [TOK_HEX + int(h) for h in VERTEX_TO_HEX[v] if h != NO_HEX])
	for h in range(N_HEXES):
		put(TOK_HEX + h, [int(v) for v in HEX_TO_VERTEX[h]])
	for p in range(N_PLAYERS):
		put(TOK_PLAYER + p, [TOK_GLOBAL])
	put(TOK_GLOBAL, [TOK_PLAYER + p for p in range(N_PLAYERS)])
	A = np.zeros((N_TOKENS, N_TOKENS), dtype=np.float32)
	for t in range(N_TOKENS):
		for k in range(MAX_NBR):
			A[t, nbr[t, k]] += wts[t, k, 0]
	return nbr, wts, A


def _build_context():
	"""Row vector c with c.g == mean(g) + g[global] + g[viewer]: one matmul
	instead of a ReduceMean, two slices and two adds."""
	c = np.full((1, N_TOKENS), 1. / N_TOKENS, dtype=np.float32)
	c[0, TOK_GLOBAL] += 1.
	c[0, TOK_PLAYER] += 1.
	return c


def _one_hot(cols, k, n_tok, n_col):
	"""(B, n_tok, n_col) float columns -> (B, n_tok, n_col*k) one-hot. Equal +
	Cast + Reshape: no round, no clamp. A value outside 0..k-1 (dummy tracing
	input) simply lights nothing, so the result is finite for any input.
	n_tok / n_col are passed as Python ints on purpose: read off the tensor
	they would be TRACED, and a dynamic-batch export then spends a
	Shape/Gather/Unsqueeze/Concat chain per call to rebuild a constant."""
	ar = torch.arange(k, dtype=cols.dtype, device=cols.device)
	oh = (cols.unsqueeze(-1) == ar).to(cols.dtype)
	return oh.reshape(-1, n_tok, n_col * k)


class CatEmbed(nn.Module):
	"""Sum of embeddings over several categorical columns of one token type, with
	a single stacked table and a single gather. `fields` is a list of (column,
	size, share_key): columns with the same share_key use the same rows (the 3
	edge slots of a vertex must share one table for equivariance)."""

	def __init__(self, fields, d):
		super().__init__()
		offsets, sizes, keys = [], [], {}
		total = 0
		for col, size, key in fields:
			if key not in keys:
				keys[key] = total
				total += size
			offsets.append(keys[key])
			sizes.append(size)
		self.table = nn.Embedding(total, d)
		self.register_buffer('cols', torch.tensor([f[0] for f in fields], dtype=torch.long), persistent=False)
		self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long), persistent=False)
		self.register_buffer('maxs', torch.tensor([s - 1 for s in sizes], dtype=torch.long), persistent=False)

	def forward(self, rows):                          # rows: (B, T, N_COLS) float
		idx = rows[:, :, self.cols].round().long().clamp(min=0)
		idx = torch.minimum(idx, self.maxs) + self.offsets
		return self.table(idx).sum(2)


class CatanNNet(nn.Module):
	def __init__(self, game, args):
		super().__init__()
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
		self.dim, self.rank = d, rank
		self.num_players = game.num_players if game is not None else N_PLAYERS
		P = self.num_players

		self.fast = bool(cfg.get('fast', False))

		nbr, wts, A = _build_neighbours()
		self.register_buffer('A', torch.from_numpy(A), persistent=False)
		self.register_buffer('ctx_w', torch.from_numpy(_build_context()), persistent=False)
		self.register_buffer('neg_inf', torch.tensor(-1e9), persistent=False)   # scalar: no full_like
		self.register_buffer('edge_u', torch.from_numpy(EDGE_TO_VERTEX[:, 0].astype(np.int64)), persistent=False)
		self.register_buffer('edge_v', torch.from_numpy(EDGE_TO_VERTEX[:, 1].astype(np.int64)), persistent=False)

		# one static class per token, all isometry-invariant: vertex (deg, land),
		# hex, player seat, global
		v_deg = (VERTEX_TO_EDGE != NO_EDGE).sum(1)
		v_land = (VERTEX_TO_HEX != NO_HEX).sum(1)
		static = np.concatenate([v_deg * 4 + v_land, np.full(N_HEXES, 16),
		                         17 + np.arange(P), [17 + P]]).astype(np.int64)
		self.register_buffer('static_idx', torch.from_numpy(static), persistent=False)
		self.emb_static = nn.Embedding(18 + P, d)

		if self.fast:
			self._build_fast(d, P, rank)
		else:
			self._build_base(d, P, rank)

		self.in_norm = nn.LayerNorm(d)
		self.lin = nn.ModuleList([nn.Linear(d, 3 * d) for _ in range(self.n_layers)])
		self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.n_layers)])

	def _build_base(self, d, P, rank):
		self.cat_vertex = CatEmbed([(V_BUILDING, 3, 'b'), (V_OWNER, P + 1, 'o'), (V_PORT, N_PORT_TYPES, 'p'),
		                            (V_EDGE0, P + 1, 'e'), (V_EDGE0 + 1, P + 1, 'e'), (V_EDGE0 + 2, P + 1, 'e')], d)
		self.cat_hex = CatEmbed([(H_TYPE, N_HEX_TYPES, 't'), (H_PIPS, 6, 'i'), (H_TOKEN, N_TOKEN_IDS, 'k'),
		                         (H_ROBBER, 2, 'r')], d)
		self.cat_player = CatEmbed([(3 * N_COLS + PD_TRADE_STATUS, 4, 's')], d)
		self.emb_masked = nn.Embedding(2, d)
		self.cat_global = CatEmbed([(GA_PHASE, N_PHASES, 'ph'), (GA_DICE, 13, 'di'),
		                            (N_COLS + GB_TURN_PLAYER, P, 'tu')], d)

		self.n_player_feat = 3 * N_RESOURCES + 3 + 7 + 3 + 6 + 2 * N_RESOURCES
		self.player_proj = nn.Linear(self.n_player_feat, d)
		self.n_global_feat = 2 * N_RESOURCES + N_DEV_TYPES + 4 + 1
		self.global_proj = nn.Linear(self.n_global_feat, d)

		# one matmul over all tokens: [settlement, city, edge_proj(rank), value score]
		self.head_tok = nn.Linear(d, 2 + rank + 1)
		self.head_edge_w = nn.Parameter(torch.randn(rank) * (rank ** -0.5))
		self.head_edge_b = nn.Parameter(torch.zeros(1))
		self.head_hex = nn.Linear(d, P)
		self.head_global = nn.Linear(d, N_GLOBAL_HEAD_A + N_GLOBAL_HEAD_B + N_GLOBAL_HEAD_C)
		self.value_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, P))

	# ---- fast family -------------------------------------------------------
	K_VERTEX = N_PORT_TYPES              # widest vertex field (port: 7 values)
	K_HEX = N_TOKEN_IDS                  # widest hex field (token id: 11 values)
	K_GLOBAL = 13                        # widest global field (dice: 0..12)

	def _build_fast(self, d, P, rank):
		K = self.K_VERTEX
		# vertex: 6 one-hot fields of width K; the 3 edge slots are TIED to one
		# weight group (an isometry permutes them), so the parameter has 4 groups
		# and a constant index expands it to 6 -- folded away at export.
		self.w_vertex = nn.Parameter(torch.randn(4 * K, d))
		expand = np.concatenate([np.arange(0, K), np.arange(K, 2 * K), np.arange(2 * K, 3 * K)]
		                        + [np.arange(3 * K, 4 * K)] * 3)
		self.register_buffer('vertex_expand', torch.from_numpy(expand.astype(np.int64)), persistent=False)
		self.w_hex = nn.Parameter(torch.randn(4 * self.K_HEX, d))

		# player: the raw 4 rows (48 ints) times a constant scale, plus a one-hot
		# of the trade status and the masked flag; one Linear over the lot.
		scale = np.zeros(4 * N_COLS, dtype=np.float32)
		scale[PA_RESOURCES:PA_RESOURCES + N_RESOURCES] = 1. / BANK_PER_RESOURCE
		scale[PA_DEV_PLAYABLE:PA_DEV_PLAYABLE + N_DEV_TYPES] = 1. / 5.
		scale[PA_TOTAL_RES] = 1. / BANK_PER_RESOURCE
		scale[PA_TOTAL_DEV] = 1. / 25.
		scale[N_COLS + PB_DEV_NEW:N_COLS + PB_DEV_NEW + N_DEV_TYPES] = 1. / 5.
		scale[N_COLS + PB_KNIGHTS] = 1. / 14.
		scale[N_COLS + PB_SETTLEMENTS_LEFT] = 1. / MAX_SETTLEMENTS
		scale[N_COLS + PB_CITIES_LEFT] = 1. / MAX_CITIES
		scale[N_COLS + PB_ROADS_LEFT] = 1. / MAX_ROADS
		scale[N_COLS + PB_ROAD_LENGTH] = 1. / MAX_ROADS
		scale[N_COLS + PB_HAS_ROAD] = scale[N_COLS + PB_HAS_ARMY] = 1.
		scale[2 * N_COLS + PC_VP_PUBLIC] = 1. / VP_TO_WIN
		scale[2 * N_COLS + PC_VP_DEV] = 1. / 5.
		scale[2 * N_COLS + PC_DEV_PLAYED_THIS_TURN] = 1.
		scale[2 * N_COLS + PC_PORTS:2 * N_COLS + PC_PORTS + 6] = 1.
		scale[2 * N_COLS + PC_TOTAL_DEV_NEW] = 1. / 5.
		scale[2 * N_COLS + PC_DISCARD_LEFT] = 1. / HAND_LIMIT_ON_SEVEN
		scale[2 * N_COLS + PC_TRADES_THIS_TURN] = 1. / MAX_TRADES_PER_TURN
		scale[3 * N_COLS + PD_TRADE_RECV:3 * N_COLS + PD_TRADE_RECV + N_RESOURCES] = 1. / 3.
		scale[3 * N_COLS + PD_TRADE_GIVE:3 * N_COLS + PD_TRADE_GIVE + N_RESOURCES] = 1. / 3.
		self.register_buffer('player_scale', torch.from_numpy(scale), persistent=False)
		# masked iff total - detail != 0, and total - detail is LINEAR in the row
		mvec = np.zeros((4 * N_COLS, 1), dtype=np.float32)
		mvec[PA_TOTAL_RES, 0] = mvec[PA_TOTAL_DEV, 0] = 1.
		mvec[PA_RESOURCES:PA_RESOURCES + N_RESOURCES, 0] = -1.
		mvec[PA_DEV_PLAYABLE:PA_DEV_PLAYABLE + N_DEV_TYPES, 0] = -1.
		mvec[N_COLS + PB_DEV_NEW:N_COLS + PB_DEV_NEW + N_DEV_TYPES, 0] = -1.
		self.register_buffer('masked_vec', torch.from_numpy(mvec), persistent=False)
		self.player_proj = nn.Linear(4 * N_COLS + 4 + 1, d)

		gscale = np.zeros(2 * N_COLS, dtype=np.float32)
		gscale[GA_BANK:GA_BANK + N_RESOURCES] = 1. / BANK_PER_RESOURCE
		gscale[GA_DEV_DECK:GA_DEV_DECK + N_DEV_TYPES] = 1. / 14.
		gscale[N_COLS + GB_ROUND_LO] = 1. / 100.
		gscale[N_COLS + GB_ROUND_HI] = 1. / (MAX_ROUNDS / 100.)
		gscale[N_COLS + GB_PENDING_COUNT] = 1. / 2.
		gscale[N_COLS + GB_SETUP_STEP] = 1. / (2 * P)
		gscale[N_COLS + GB_DEV_PLAYED:N_COLS + GB_DEV_PLAYED + N_DEV_TYPES] = 1. / 14.
		gscale[N_COLS + GB_CHANCE_COUNTER] = 1. / 100.
		gscale[N_COLS + GB_PLAYER_TRADE_DONE] = 1.
		self.register_buffer('global_scale', torch.from_numpy(gscale), persistent=False)
		self.register_buffer('global_cat_cols', torch.tensor([GA_PHASE, GA_DICE, N_COLS + GB_TURN_PLAYER]),
		                     persistent=False)
		self.global_proj = nn.Linear(2 * N_COLS + 3 * self.K_GLOBAL, d)

		self.head_tok = nn.Linear(d, 2 + rank)                 # vertices only
		self.head_edge_w = nn.Parameter(torch.randn(rank) * (rank ** -0.5))
		self.head_edge_b = nn.Parameter(torch.zeros(1))
		self.head_hex = nn.Linear(d, P)
		self.head_global = nn.Linear(d, N_GLOBAL_HEAD_A + N_GLOBAL_HEAD_B + N_GLOBAL_HEAD_C)
		self.value_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, P))

	def _encode_fast(self, board):
		P = self.num_players
		verts = board[:, :N_VERTICES, :V_EDGE2 + 1]
		hexes = board[:, TOK_HEX:TOK_HEX + N_HEXES, :H_ROBBER + 1]
		players = board[:, TOK_PLAYER:TOK_PLAYER + 4 * P, :].reshape(-1, P, 4 * N_COLS)
		glob = board[:, TOK_PLAYER + 4 * P:, :].reshape(-1, 1, 2 * N_COLS)

		hv = torch.matmul(_one_hot(verts, self.K_VERTEX, N_VERTICES, V_EDGE2 + 1), self.w_vertex[self.vertex_expand])
		hh = torch.matmul(_one_hot(hexes, self.K_HEX, N_HEXES, H_ROBBER + 1), self.w_hex)

		masked = (torch.matmul(players, self.masked_vec) > 0.5).to(board.dtype)
		status = _one_hot(players[:, :, 3 * N_COLS + PD_TRADE_STATUS:3 * N_COLS + PD_TRADE_STATUS + 1], 4, P, 1)
		hp = self.player_proj(torch.cat([players * self.player_scale, status, masked], dim=-1))

		gcat = _one_hot(glob[:, :, self.global_cat_cols], self.K_GLOBAL, 1, 3)
		hg = self.global_proj(torch.cat([glob * self.global_scale, gcat], dim=-1))

		h = torch.cat([hv, hh, hp, hg], dim=1) + self.emb_static(self.static_idx).unsqueeze(0)
		return self.in_norm(h)

	def _heads_fast(self, h, valid_actions):
		tok = self.head_tok(h[:, :N_VERTICES, :])                  # (B, 54, 2 + rank)
		v_logit = tok[:, :, :2]
		pr = tok[:, :, 2:]
		e_logit = ((pr[:, self.edge_u, :] * pr[:, self.edge_v, :]) * self.head_edge_w).sum(-1) + self.head_edge_b
		h_logit = self.head_hex(h[:, TOK_HEX:TOK_HEX + N_HEXES, :]).reshape(-1, N_HEXES * self.num_players)
		hg = h[:, TOK_GLOBAL, :]
		ga, gbc = torch.split(self.head_global(hg), [N_GLOBAL_HEAD_A, N_GLOBAL_HEAD_B + N_GLOBAL_HEAD_C], dim=-1)
		pi = torch.cat([e_logit, v_logit[:, :, 0], v_logit[:, :, 1], ga, h_logit, gbc], dim=-1)
		pi = torch.where(valid_actions, pi, self.neg_inf)
		v = torch.tanh(self.value_mlp(torch.cat([hg, h.mean(dim=1)], dim=-1)))
		return F.log_softmax(pi, dim=-1), v

	def _encode(self, board):
		if self.fast:
			return self._encode_fast(board)
		P = self.num_players
		verts = board[:, :N_VERTICES, :]
		hexes = board[:, TOK_HEX:TOK_HEX + N_HEXES, :]
		players = board[:, TOK_PLAYER:TOK_PLAYER + 4 * P, :].reshape(-1, P, 4 * N_COLS)
		glob = board[:, TOK_PLAYER + 4 * P:, :].reshape(-1, 1, 2 * N_COLS)

		hv = self.cat_vertex(verts)
		hh = self.cat_hex(hexes)

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
		t_recv = players[:, :, 3 * N_COLS + PD_TRADE_RECV:3 * N_COLS + PD_TRADE_RECV + N_RESOURCES] / 3.
		t_give = players[:, :, 3 * N_COLS + PD_TRADE_GIVE:3 * N_COLS + PD_TRADE_GIVE + N_RESOURCES] / 3.
		pf = torch.cat([res, devp, devn, tot, misc, vp, ports, t_recv, t_give], dim=-1)
		detail_r = players[:, :, PA_RESOURCES:PA_RESOURCES + N_RESOURCES].sum(-1)
		detail_d = (players[:, :, PA_DEV_PLAYABLE:PA_DEV_PLAYABLE + N_DEV_TYPES].sum(-1)
		            + players[:, :, N_COLS + PB_DEV_NEW:N_COLS + PB_DEV_NEW + N_DEV_TYPES].sum(-1))
		masked = (((detail_r - players[:, :, PA_TOTAL_RES]).abs()
		           + (detail_d - players[:, :, PA_TOTAL_DEV]).abs()) > 0.5).long()
		hp = self.player_proj(pf) + self.cat_player(players) + self.emb_masked(masked)

		rnd = (glob[:, :, N_COLS + GB_ROUND_HI] * 100. + glob[:, :, N_COLS + GB_ROUND_LO]) / float(MAX_ROUNDS)
		gf = torch.cat([
			glob[:, :, GA_BANK:GA_BANK + N_RESOURCES] / float(BANK_PER_RESOURCE),
			glob[:, :, GA_DEV_DECK:GA_DEV_DECK + N_DEV_TYPES] / 14.,
			glob[:, :, N_COLS + GB_DEV_PLAYED:N_COLS + GB_DEV_PLAYED + N_DEV_TYPES] / 14.,
			torch.stack([rnd,
			             glob[:, :, N_COLS + GB_PENDING_COUNT] / 2.,
			             glob[:, :, N_COLS + GB_SETUP_STEP] / float(2 * P),
			             glob[:, :, N_COLS + GB_CHANCE_COUNTER] / 100.,
			             glob[:, :, N_COLS + GB_PLAYER_TRADE_DONE]], dim=-1),
		], dim=-1)
		hg = self.global_proj(gf) + self.cat_global(glob)

		h = torch.cat([hv, hh, hp, hg], dim=1) + self.emb_static(self.static_idx).unsqueeze(0)
		return self.in_norm(h)

	def _trunk(self, h):
		for i in range(self.n_layers):
			s, n, g = torch.split(self.lin[i](h), self.dim, dim=-1)
			agg = torch.matmul(self.A, n)                          # mean over neighbours
			ctx = torch.matmul(self.ctx_w, g)                     # mean(all) + global + viewer
			h = self.norm[i](h + F.gelu(s + agg + ctx))
		return h

	def forward(self, board, valid_actions):
		h = self._trunk(self._encode(board))
		if self.fast:
			return self._heads_fast(h, valid_actions)
		tok = self.head_tok(h)                                     # (B, T, 2 + rank + 1)
		v_logit = tok[:, :N_VERTICES, :2]
		pr = tok[:, :N_VERTICES, 2:2 + self.rank]
		e_logit = ((pr[:, self.edge_u, :] * pr[:, self.edge_v, :]) * self.head_edge_w).sum(-1) + self.head_edge_b
		h_logit = self.head_hex(h[:, TOK_HEX:TOK_HEX + N_HEXES, :]).reshape(-1, N_HEXES * self.num_players)
		ga, gb, gc = torch.split(self.head_global(h[:, TOK_GLOBAL, :]),
		                         [N_GLOBAL_HEAD_A, N_GLOBAL_HEAD_B, N_GLOBAL_HEAD_C], dim=-1)
		pi = torch.cat([e_logit, v_logit[:, :, 0], v_logit[:, :, 1], ga, h_logit, gb, gc], dim=-1)
		pi = torch.where(valid_actions, pi, self.neg_inf)
		w = torch.softmax(tok[:, :, 2 + self.rank], dim=-1)
		pooled = torch.einsum('bt,btd->bd', w, h)
		v = torch.tanh(self.value_mlp(pooled))
		return F.log_softmax(pi, dim=-1), v
