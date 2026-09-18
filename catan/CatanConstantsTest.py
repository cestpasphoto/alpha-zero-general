"""Assertions on CatanConstants. Run this before writing a single line of game logic.

Everything checked here is structural: it cannot be checked later by playing games,
because a wrong incidence table produces a board that is perfectly self-consistent
and simply wrong.
"""
import numpy as np
from CatanConstants import *


def check_material():
	assert HEX_DISTRIBUTION.size == N_HEXES
	assert (HEX_DISTRIBUTION == HEX_DESERT).sum() == 1
	assert TOKEN_DISTRIBUTION.size == N_HEXES - 1
	assert TOKEN_VALUES.size == N_TOKEN_IDS and TOKEN_PIPS.size == N_TOKEN_IDS
	for t in range(1, N_TOKEN_IDS):
		assert TOKEN_PIPS[t] == 6 - abs(7 - TOKEN_VALUES[t]), t
	assert TOKEN_PIPS[0] == 0 and TOKEN_VALUES[0] == 0
	assert (TOKEN_VALUES[1:] != 7).all()
	# 2 and 12 appear once, every other token twice -> 58 pips on the board
	assert np.bincount(TOKEN_DISTRIBUTION, minlength=N_TOKEN_IDS).tolist() == [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1]
	assert TOKEN_PIPS[TOKEN_DISTRIBUTION].sum() == 58
	assert DEV_DISTRIBUTION.sum() == 25 and DEV_DISTRIBUTION.size == N_DEV_TYPES
	assert PORT_DISTRIBUTION.size == N_PORTS
	assert sorted(PORT_DISTRIBUTION[PORT_DISTRIBUTION != PORT_GENERIC]) == [1, 2, 3, 4, 5]
	assert (HEX_TO_RESOURCE[1:] == np.arange(N_RESOURCES)).all()
	assert HEX_TO_RESOURCE[HEX_DESERT] == NO_RESOURCE
	for c in (COST_ROAD, COST_SETTLEMENT, COST_CITY, COST_DEV):
		assert c.size == N_RESOURCES and (c >= 0).all()
	assert YOP_PAIRS.shape == (15, 2)
	print("material                       OK")


def check_topology():
	assert VERTEX_TO_HEX.shape == (54, 3) and EDGE_TO_VERTEX.shape == (72, 2)
	assert HEX_TO_VERTEX.shape == (19, 6) and HEX_TO_EDGE.shape == (19, 6)

	# every hex sees 6 distinct vertices and 6 distinct edges
	for h in range(N_HEXES):
		assert len(set(HEX_TO_VERTEX[h])) == 6 and len(set(HEX_TO_EDGE[h])) == 6, h
	# every vertex/edge is covered
	assert set(HEX_TO_VERTEX.flatten()) == set(range(N_VERTICES))
	assert set(HEX_TO_EDGE.flatten()) == set(range(N_EDGES))
	# degrees
	deg_e = (VERTEX_TO_EDGE != NO_EDGE).sum(1)
	assert set(deg_e.tolist()) == {2, 3} and (deg_e == 3).sum() == 36 and (deg_e == 2).sum() == 18
	assert ((VERTEX_TO_HEX != NO_HEX).sum(1) >= 1).all()
	assert (VERTEX_TO_HEX != NO_HEX).sum() == 6 * N_HEXES
	# v2e and v2v are consistent, and the relation is symmetric
	for v in range(N_VERTICES):
		for k in range(3):
			e, o = VERTEX_TO_EDGE[v, k], VERTEX_TO_VERTEX[v, k]
			assert (e == NO_EDGE) == (o == NO_VERTEX), (v, k)
			if e != NO_EDGE:
				assert sorted(EDGE_TO_VERTEX[e]) == sorted([v, o]), (v, k)
				assert v in VERTEX_TO_VERTEX[o], (v, k)
	# a vertex and a hex are incident iff the hex lists the vertex
	for v in range(N_VERTICES):
		for h in VERTEX_TO_HEX[v]:
			if h != NO_HEX:
				assert v in HEX_TO_VERTEX[h], (v, h)
	# each edge of a hex joins two vertices of that same hex
	for h in range(N_HEXES):
		for e in HEX_TO_EDGE[h]:
			assert set(EDGE_TO_VERTEX[e]) <= set(HEX_TO_VERTEX[h]), (h, e)
	for e in range(N_EDGES):
		for i in range(2):
			assert VERTEX_TO_EDGE[EDGE_TO_VERTEX[e, i], EDGE_SLOT[e, i]] == e, (e, i)
	assert COASTAL_EDGES.size == 30
	print("topology (54/72/19, degrees)   OK")


def check_ports():
	assert PORT_EDGES.size == N_PORTS and PORT_VERTICES.shape == (N_PORTS, 2)
	assert len(set(PORT_EDGES.tolist())) == N_PORTS
	assert set(PORT_EDGES.tolist()) <= set(COASTAL_EDGES.tolist())
	flat = PORT_VERTICES.flatten().tolist()
	assert len(set(flat)) == 2 * N_PORTS, "two ports share a vertex"
	for p in range(N_PORTS):
		assert sorted(EDGE_TO_VERTEX[PORT_EDGES[p]]) == sorted(PORT_VERTICES[p]), p
	print("ports (9 disjoint coastal)     OK")


def check_isometries():
	ident = np.arange(N_VERTICES, dtype=np.int8)
	assert (ISO_VERTEX[0] == ident).all(), "isometry 0 must be the identity"
	seen = set()
	for s in range(N_ISOMETRIES):
		for tab, n in ((ISO_HEX[s], N_HEXES), (ISO_VERTEX[s], N_VERTICES), (ISO_EDGE[s], N_EDGES)):
			assert sorted(tab.tolist()) == list(range(n)), (s, n)
		seen.add(ISO_VERTEX[s].tobytes())
		# incidences must commute with the isometry, otherwise the augmented
		# (state, policy) pair is silently inconsistent
		for e in range(N_EDGES):
			assert sorted(ISO_VERTEX[s][EDGE_TO_VERTEX[e]]) == sorted(EDGE_TO_VERTEX[ISO_EDGE[s][e]]), (s, e)
		for h in range(N_HEXES):
			assert sorted(ISO_VERTEX[s][HEX_TO_VERTEX[h]]) == sorted(HEX_TO_VERTEX[ISO_HEX[s][h]]), (s, h)
			assert sorted(ISO_EDGE[s][HEX_TO_EDGE[h]]) == sorted(HEX_TO_EDGE[ISO_HEX[s][h]]), (s, h)
	assert len(seen) == N_ISOMETRIES, "the 12 isometries are not distinct"
	# closure: the set is a group under composition
	comp = set()
	for a in range(N_ISOMETRIES):
		for b in range(N_ISOMETRIES):
			comp.add(ISO_VERTEX[a][ISO_VERTEX[b]].tobytes())
	assert comp == seen, "isometries are not closed under composition"
	print("12 isometries (group, commute) OK")


def check_message_passing():
	assert MP_VERTEX_VERTEX.shape == (2 * N_EDGES, 2)
	assert MP_VERTEX_HEX.shape == (6 * N_HEXES, 2)
	for arr, hi in ((MP_VERTEX_VERTEX, N_VERTICES), (MP_VERTEX_HEX, N_HEXES)):
		assert (arr >= 0).all() and (arr[:, 1] < hi).all() and (arr[:, 0] < N_VERTICES).all()
	print("message-passing lists          OK")


def check_actions():
	blocks = [("road", A_ROAD, N_EDGES), ("settlement", A_SETTLEMENT, N_VERTICES),
	          ("city", A_CITY, N_VERTICES), ("buy_dev", A_BUY_DEV, 1),
	          ("play_dev", A_PLAY_DEV, 2), ("robber", A_ROBBER, N_HEXES * N_PLAYERS), ("roll", A_ROLL, 1),
	          ("monopoly", A_MONOPOLY, N_RESOURCES), ("yop", A_YEAR_OF_PLENTY, 15),
	          ("bank_trade", A_BANK_TRADE, 20), ("discard", A_DISCARD, N_RESOURCES),
	          ("end_turn", A_END_TURN, 1),
	          ("trade_recv", A_TRADE_RECV, N_TRADE_SETS), ("trade_give", A_TRADE_GIVE, N_TRADE_SETS),
	          ("trade_ok", A_TRADE_OK, 1), ("trade_no", A_TRADE_NO, 1),
	          ("trade_accept", A_TRADE_ACCEPT, N_PLAYERS)]
	covered = np.zeros(N_ACTIONS, np.int8)
	for name, off, size in blocks:
		assert off >= 0 and off + size <= N_ACTIONS, name
		covered[off:off + size] += 1
	assert (covered == 1).all(), "action ranges overlap or leave a hole"
	assert N_ACTIONS_V1 == 230 + 19 * N_PLAYERS
	assert N_ACTIONS == N_ACTIONS_V1 + 2 * N_TRADE_SETS + 2 + N_PLAYERS

	# The 55 trade multisets: distinct, sized 1..3, and covering every such multiset
	from itertools import combinations_with_replacement
	assert TRADE_SETS.shape == (N_TRADE_SETS, N_RESOURCES)
	assert (TRADE_SETS >= 0).all() and (TRADE_SETS.sum(1) >= 1).all() and (TRADE_SETS.sum(1) <= 3).all()
	assert len({tuple(r) for r in TRADE_SETS}) == N_TRADE_SETS, "duplicate trade multiset"
	ref = set()
	for size in (1, 2, 3):
		for combo in combinations_with_replacement(range(N_RESOURCES), size):
			c = [0] * N_RESOURCES
			for r in combo:
				c[r] += 1
			ref.add(tuple(c))
	assert {tuple(r) for r in TRADE_SETS} == ref, "TRADE_SETS is not exactly the 1..3 multisets"

	# The trade ids name RESOURCES, which no isometry permutes, so every one of
	# them must map to ITSELF. A silently permuted trade id would corrupt the x12
	# augmentation and the policy target with it -- and nothing downstream would
	# say so, which is exactly why this is asserted rather than assumed.
	for s in range(N_ISOMETRIES):
		for a in range(N_ACTIONS_V1, N_ACTIONS):
			assert ISO_ACTION[s, a] == a, f"isometry {s} moves trade action {a}"
	print(f"actions: v1={N_ACTIONS_V1}, trade={N_ACTIONS - N_ACTIONS_V1}, total={N_ACTIONS}  OK")


def check_state_layout():
	assert N_ROWS == 75 + 4 * N_PLAYERS
	for name, cols in (("vertex", [V_BUILDING, V_OWNER, V_PORT, V_EDGE0, V_EDGE1, V_EDGE2]),
	                   ("hex", [H_TYPE, H_PIPS, H_TOKEN, H_ROBBER]),
	                   ("playerA", [PA_RESOURCES + 4, PA_DEV_PLAYABLE + 4, PA_TOTAL_RES, PA_TOTAL_DEV]),
	                   ("playerB", [PB_DEV_NEW + 4, PB_KNIGHTS, PB_ROADS_LEFT, PB_HAS_ARMY]),
	                   ("playerC", [PC_VP_PUBLIC, PC_VP_DEV, PC_DEV_PLAYED_THIS_TURN, PC_PORTS + 5, PC_TOTAL_DEV_NEW, PC_DISCARD_LEFT, PC_TRADES_THIS_TURN]),
	                   ("playerD", [PD_TRADE_RECV + 4, PD_TRADE_GIVE + 4, PD_TRADE_STATUS]),
	                   ("globalA", [GA_BANK + 4, GA_DEV_DECK + 4, GA_DICE, GA_PHASE]),
	                   ("globalB", [GB_ROUND_LO, GB_ROUND_HI, GB_PENDING_COUNT, GB_SETUP_STEP, GB_DEV_PLAYED + 4, GB_CHANCE_COUNTER, GB_TURN_PLAYER, GB_PLAYER_TRADE_DONE])):
		assert max(cols) < N_COLS, name
	assert ROWS_PER_PLAYER == 4 and ROW_GLOBAL == ROW_PLAYER + 4 * N_PLAYERS
	# row D must not collide with row C's columns: they are different rows, but a
	# mistyped PD_* would silently alias a PC_* and the battery would not notice
	assert PD_TRADE_GIVE == PD_TRADE_RECV + N_RESOURCES
	assert PD_TRADE_STATUS >= PD_TRADE_GIVE + N_RESOURCES
	assert len({TRADE_NONE, TRADE_COMPOSING, TRADE_OFFERED, TRADE_REFUSED}) == 4
	assert 2 <= N_PLAYERS <= 4
	print(f"state: ({N_ROWS}, {N_COLS}) = {N_ROWS * N_COLS} bytes, {N_VERTICES + N_HEXES + N_PLAYERS + 1} NN tokens  OK")


if __name__ == '__main__':
	check_material()
	check_topology()
	check_ports()
	check_isometries()
	check_message_passing()
	check_actions()
	check_state_layout()
	print("\nALL CONSTANT CHECKS PASSED")
