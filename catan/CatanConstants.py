import numpy as np

############################## GAME CONFIGURATION #############################

N_PLAYERS = 3                  # 2, 3 or 4 -- changes observation_size() and action_size()
ENABLE_PLAYER_TRADE = True     # player-to-player trade; see the TRADE section below
RANDOM_BOARD = True            # shuffle hexes, number tokens and port types at init
FORBID_ADJACENT_RED = True     # official rule: no two 6/8 on adjacent hexes

############################## MATERIAL #######################################

# Resource ids, used as INDICES in every 5-slot array (hands, bank, costs)
BRICK, LUMBER, ORE, GRAIN, WOOL = 0, 1, 2, 3, 4
N_RESOURCES = 5
NO_RESOURCE = 5                # sentinel, never negative (see "no negative values" rule)

# Hex categories, stored as such in the state (0 = desert, so "empty" is 0)
HEX_DESERT, HEX_HILL, HEX_FOREST, HEX_MOUNTAIN, HEX_FIELD, HEX_PASTURE = 0, 1, 2, 3, 4, 5
N_HEX_TYPES = 6
HEX_TO_RESOURCE = np.array([NO_RESOURCE, BRICK, LUMBER, ORE, GRAIN, WOOL], dtype=np.int8)
HEX_DISTRIBUTION = np.array([HEX_DESERT] * 1 + [HEX_HILL] * 3 + [HEX_FOREST] * 4 +
                            [HEX_MOUNTAIN] * 3 + [HEX_FIELD] * 4 + [HEX_PASTURE] * 4, dtype=np.int8)

# Number tokens. TOKEN_ID is the CATEGORICAL id (0 = desert, 1..10 = 2,3,4,5,6,8,9,10,11,12),
# PIPS is the number of 2d6 combinations, i.e. 36*P(roll) -- the SCALAR the network needs.
# Both are stored: pips carry the economic value, the token id carries the identity
# (two hexes on 8 are perfectly correlated, an 6 and an 8 are not).
TOKEN_VALUES = np.array([0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12], dtype=np.int8)   # indexed by token id
TOKEN_PIPS = np.array([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1], dtype=np.int8)        # indexed by token id
TOKEN_DISTRIBUTION = np.array([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10], dtype=np.int8)
RED_TOKEN_IDS = np.array([5, 6], dtype=np.int8)   # 6 and 8
N_TOKEN_IDS = 11

# Development cards
KNIGHT, VICTORY_POINT, ROAD_BUILDING, MONOPOLY, YEAR_OF_PLENTY = 0, 1, 2, 3, 4
N_DEV_TYPES = 5
DEV_DISTRIBUTION = np.array([14, 5, 2, 2, 2], dtype=np.int8)   # 25 cards

# Ports
PORT_NONE, PORT_GENERIC = 0, 6     # 1..5 = 2:1 port on resource id 0..4
N_PORT_TYPES = 7
PORT_DISTRIBUTION = np.array([PORT_GENERIC] * 4 + [1, 2, 3, 4, 5], dtype=np.int8)   # 9 ports
PORT_RATIO_GENERIC, PORT_RATIO_SPECIFIC, BANK_RATIO = 3, 2, 4

# Costs, indexed by resource id
COST_ROAD = np.array([1, 1, 0, 0, 0], dtype=np.int8)
COST_SETTLEMENT = np.array([1, 1, 0, 1, 1], dtype=np.int8)
COST_CITY = np.array([0, 0, 3, 2, 0], dtype=np.int8)
COST_DEV = np.array([0, 0, 1, 1, 1], dtype=np.int8)

# Limits
BANK_PER_RESOURCE = 19
MAX_SETTLEMENTS, MAX_CITIES, MAX_ROADS = 5, 4, 15
VP_TO_WIN = 10
MIN_KNIGHTS_FOR_ARMY = 3
MIN_LENGTH_FOR_ROAD = 5
HAND_LIMIT_ON_SEVEN = 7
MAX_ROUNDS = 400               # draw guard; stored as (lo, hi) to stay within int8
# NOT an official rule: bank trades are the one action type with no natural cap
# (roads/settlements/cities are limited by lifetime piece counts, dev buys by
# the 25-card deck shared by everyone) -- from a maxed-out hand (19 of a
# resource, the bank's own ceiling) a single turn can fit up to 41 consecutive
# 2:1 trades. Sustained across MAX_ROUNDS that is enough to push MCTS's
# recursive search() (one Python stack frame per action) past tens of
# thousands of frames on a single simulation. This caps it at the rules level,
# which is more sessions than any sensible policy would ever need in one turn.
MAX_TRADES_PER_TURN = 6

############################## TOPOLOGY #######################################
#
# Built here rather than pasted as literals: 54*3 + 72*2 + 19*6 + 12*145 numbers
# cannot be proof-read, a constructor plus assertions can. Cube coordinates
# (x+y+z=0); the land board is the radius-2 hexagon (19 hexes). A VERTEX is the
# unordered triple of hexes meeting at a corner, an EDGE the unordered pair of
# hexes sharing a side -- both are then pure set operations, no geometry.
#
# Ordering is reading order (top to bottom, left to right) for all three families.

N_HEXES, N_VERTICES, N_EDGES, N_PORTS = 19, 54, 72, 9

_DIRS = [(1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1)]   # cyclic


def _add(a, b):
	return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _pos(h):
	# integer "doubled" pointy-top position, used only to sort in reading order
	return (2 * h[0] + h[2], 3 * h[2])


def _build_topology():
	land = [(x, y, -x - y) for x in range(-2, 3) for y in range(-2, 3) if abs(x + y) <= 2]
	landset = set(land)
	verts, edges = set(), set()
	for h in land:
		for i in range(6):
			verts.add(tuple(sorted([h, _add(h, _DIRS[i]), _add(h, _DIRS[(i + 1) % 6])])))
			edges.add(tuple(sorted([h, _add(h, _DIRS[i])])))
	vpos = lambda t: tuple(sum(_pos(h)[i] for h in t) for i in (0, 1))
	epos = lambda e: tuple(sum(_pos(h)[i] for h in e) for i in (0, 1))
	hexes = sorted(land, key=lambda h: (_pos(h)[1], _pos(h)[0]))
	verts = sorted(verts, key=lambda t: (vpos(t)[1], vpos(t)[0]))
	edges = sorted(edges, key=lambda e: (epos(e)[1], epos(e)[0]))
	HI = {h: i for i, h in enumerate(hexes)}
	VI = {v: i for i, v in enumerate(verts)}
	EI = {e: i for i, e in enumerate(edges)}

	# NO_HEX / NO_EDGE / NO_VERTEX are used instead of -1: coastal vertices have
	# only 1 or 2 land hexes, and 2 or 3 edges.
	v2h = np.full((N_VERTICES, 3), N_HEXES, np.int8)
	v2e = np.full((N_VERTICES, 3), N_EDGES, np.int8)
	v2v = np.full((N_VERTICES, 3), N_VERTICES, np.int8)
	e2v = np.zeros((N_EDGES, 2), np.int8)
	h2v = np.zeros((N_HEXES, 6), np.int8)
	h2e = np.zeros((N_HEXES, 6), np.int8)

	for v, tri in enumerate(verts):
		ls = sorted(HI[h] for h in tri if h in landset)
		v2h[v, :len(ls)] = ls
	for e, (a, b) in enumerate(edges):
		ends = sorted(VI[t] for t in verts if a in t and b in t)
		e2v[e] = ends
		for u, o in ((ends[0], ends[1]), (ends[1], ends[0])):
			k = int((v2e[u] < N_EDGES).sum())
			v2e[u, k], v2v[u, k] = e, o
	for h in land:
		for i in range(6):
			h2v[HI[h], i] = VI[tuple(sorted([h, _add(h, _DIRS[i]), _add(h, _DIRS[(i + 1) % 6])]))]
			h2e[HI[h], i] = EI[tuple(sorted([h, _add(h, _DIRS[i])]))]

	# 12 isometries of the hexagon: 6 rotations x mirror. Rotation and reflection
	# are linear in cube coordinates, so vertices and edges follow for free.
	rot = lambda h: (-h[2], -h[0], -h[1])
	refl = lambda h: (h[0], h[2], h[1])
	iso_h = np.zeros((12, N_HEXES), np.int8)
	iso_v = np.zeros((12, N_VERTICES), np.int8)
	iso_e = np.zeros((12, N_EDGES), np.int8)
	for s, (m, k) in enumerate([(m, k) for m in (0, 1) for k in range(6)]):
		def g(h, m=m, k=k):
			c = refl(h) if m else h
			for _ in range(k):
				c = rot(c)
			return c
		for h in land:
			iso_h[s, HI[h]] = HI[g(h)]
		for v, tri in enumerate(verts):
			iso_v[s, v] = VI[tuple(sorted(map(g, tri)))]
		for e, pair in enumerate(edges):
			iso_e[s, e] = EI[tuple(sorted(map(g, pair)))]

	# Ports sit on 9 of the 30 coastal edges, spaced 3-3-4 all the way round.
	# Positions belong to the FRAME and never change; only the types are shuffled.
	import math
	coastal = [e for e, pair in enumerate(edges) if sum(1 for h in pair if h in landset) == 1]
	ang = lambda e: math.atan2(epos(edges[e])[1] / math.sqrt(3.), epos(edges[e])[0])
	ring = sorted(coastal, key=ang)
	idx, chosen = 0, []
	for gap in [3, 3, 4] * 3:
		chosen.append(ring[idx % len(ring)])
		idx += gap
	port_e = np.array(sorted(chosen), np.int8)
	port_v = np.array([sorted(e2v[e]) for e in sorted(chosen)], np.int8)

	# slot of edge e inside each of its two endpoints
	eslot = np.zeros((N_EDGES, 2), np.int8)
	for e in range(N_EDGES):
		for i in range(2):
			eslot[e, i] = int(np.flatnonzero(v2e[e2v[e, i]] == e)[0])

	# hex neighbours, through a shared edge
	h2h = np.full((N_HEXES, 6), N_HEXES, np.int8)
	for h in land:
		k = 0
		for i in range(6):
			n = _add(h, _DIRS[i])
			if n in landset:
				h2h[HI[h], k] = HI[n]
				k += 1

	# slot of edge ISO_EDGE[s,e] inside the image vertex, precomputed so that
	# get_symmetries never searches at runtime
	iso_slot = np.zeros((12, N_VERTICES, 3), np.int8)
	for s in range(12):
		for v in range(N_VERTICES):
			for k in range(3):
				e = v2e[v, k]
				if e == N_EDGES:
					continue
				nv, ne = iso_v[s, v], iso_e[s, e]
				iso_slot[s, v, k] = int(np.flatnonzero(v2e[nv] == ne)[0])

	# Integer layout coordinates, for CatanDisplay only. Doubled pointy-top
	# coordinates: a hex centre is (2q+r, 3r), a vertex/edge the sum over the
	# hexes that define it, so everything stays exact.
	hpos = np.array([_pos(h) for h in hexes], np.int16)
	vpos_a = np.array([vpos(t) for t in verts], np.int16)
	epos_a = np.array([epos(e) for e in edges], np.int16)

	return (v2h, v2e, v2v, e2v, eslot, h2v, h2e, h2h, iso_h, iso_v, iso_e, iso_slot,
	        port_e, port_v, np.array(coastal, np.int8), hpos, vpos_a, epos_a)


NO_HEX, NO_EDGE, NO_VERTEX = N_HEXES, N_EDGES, N_VERTICES

(VERTEX_TO_HEX, VERTEX_TO_EDGE, VERTEX_TO_VERTEX, EDGE_TO_VERTEX, EDGE_SLOT,
 HEX_TO_VERTEX, HEX_TO_EDGE, HEX_TO_HEX, ISO_HEX, ISO_VERTEX, ISO_EDGE, ISO_SLOT,
 PORT_EDGES, PORT_VERTICES, COASTAL_EDGES, HEX_POS, VERTEX_POS, EDGE_POS) = _build_topology()

N_ISOMETRIES = 12

# Message-passing index lists for the network: plain (src, dst) pairs, no sentinel,
# so the encoder never does arithmetic on a "missing" marker.
MP_VERTEX_VERTEX = np.array([[EDGE_TO_VERTEX[e, i], EDGE_TO_VERTEX[e, 1 - i]]
                             for e in range(N_EDGES) for i in (0, 1)], dtype=np.int8)   # 144 pairs
MP_VERTEX_HEX = np.array([[v, VERTEX_TO_HEX[v, k]]
                          for v in range(N_VERTICES) for k in range(3)
                          if VERTEX_TO_HEX[v, k] != NO_HEX], dtype=np.int8)             # 114 pairs

############################## STATE LAYOUT ###################################
#
# state = int8 array of shape (75 + 4*N_PLAYERS, 12). See the header of
# CatanLogicNumba.py for the column-by-column description.

N_COLS = 12
ROWS_PER_PLAYER = 4                             # A hand / B pieces / C public / D trade
ROW_VERTEX = 0                                  # 54 rows, one per vertex  -> NN token
ROW_HEX = ROW_VERTEX + N_VERTICES               # 19 rows, one per hex     -> NN token
ROW_PLAYER = ROW_HEX + N_HEXES                  # 4 rows per player        -> NN token
ROW_GLOBAL = ROW_PLAYER + ROWS_PER_PLAYER * N_PLAYERS   # 2 rows           -> NN token
N_ROWS = ROW_GLOBAL + 2

# Vertex columns
V_BUILDING, V_OWNER, V_PORT = 0, 1, 2           # building: 0 none / 1 settlement / 2 city
V_EDGE0, V_EDGE1, V_EDGE2 = 3, 4, 5             # owner of VERTEX_TO_EDGE[v,k], 0 = free
# Hex columns
H_TYPE, H_PIPS, H_TOKEN, H_ROBBER = 0, 1, 2, 3
# Player row A
PA_RESOURCES, PA_DEV_PLAYABLE, PA_TOTAL_RES, PA_TOTAL_DEV = 0, 5, 10, 11
# Player row B
PB_DEV_NEW, PB_KNIGHTS, PB_SETTLEMENTS_LEFT, PB_CITIES_LEFT = 0, 5, 6, 7
PB_ROADS_LEFT, PB_ROAD_LENGTH, PB_HAS_ROAD, PB_HAS_ARMY = 8, 9, 10, 11
# Player row C
PC_VP_PUBLIC, PC_VP_DEV, PC_DEV_PLAYED_THIS_TURN, PC_PORTS = 0, 1, 2, 3   # PC_PORTS: 6 slots
PC_DISCARD_LEFT = 10           # cards this player still owes after a 7
PC_TOTAL_DEV_NEW = 9            # how many dev cards were bought this turn. Public (everyone
                               # saw the purchase), only the TYPE is hidden, so masking must
                               # keep it: without it sample_world cannot tell a playable card
                               # from one bought this turn.
PC_TRADES_THIS_TURN = 11       # bank trades made this turn, reset in _start_turn(). See
                               # MAX_TRADES_PER_TURN above for why this exists.
# Player row D -- this player's standing player-trade offer, PUBLIC (an announcement
# is heard by everyone, so get_observation leaves this row alone). Keeping the offer
# on its AUTHOR's rows is what makes swap_players free: nothing here names a player,
# so the offer rotates with its owner and needs no relabelling.
PD_TRADE_RECV = 0              # 5 slots: what this player asks FOR
PD_TRADE_GIVE = 5              # 5 slots: what this player offers IN EXCHANGE
PD_TRADE_STATUS = 10           # TRADE_* below
# col 11 free

# PD_TRADE_STATUS values. The whole protocol state is DERIVED from these plus
# GB_TURN_PLAYER -- no separate "who answers next" counter to keep in sync:
#   the player composing an offer  = the unique TRADE_COMPOSING
#   the next player to answer      = first TRADE_NONE after the turn player, in seat order
TRADE_NONE, TRADE_COMPOSING, TRADE_OFFERED, TRADE_REFUSED = 0, 1, 2, 3

# Global row A
GA_BANK, GA_DEV_DECK, GA_DICE, GA_PHASE = 0, 5, 10, 11
# Global row B
GB_ROUND_LO, GB_ROUND_HI, GB_PENDING_COUNT = 0, 1, 2
GB_TURN_PLAYER = 3             # whose TURN it is, relative to the player to move. During a
                               # discard the actor is not the turn owner, hence two fields.
GB_SETUP_STEP = 4              # index in the snake placement order
GB_DEV_PLAYED = 5              # 5 slots: dev cards played and discarded, per type.
                               # Without it, conservation of non-knight dev cards is
                               # uncheckable: a played Monopoly leaves the hand and
                               # vanishes. Added because the assertion battery needed it.
GB_CHANCE_COUNTER = 10         # bumped at every chance draw, mod 100: decorrelates the
                               # successive draws of one stream (see Stochastic.py)
GB_PLAYER_TRADE_DONE = 11      # 0/1, one player-trade ATTEMPT per turn (success or not),
                               # reset in _start_turn(). Same reasoning as
                               # MAX_TRADES_PER_TURN: without it an agent can reopen a
                               # refused announcement indefinitely and regrow the depth
                               # the cap exists to bound.
# A vertex index was stored in col 11 at first; it broke isometry equivariance (a
# global scalar holding a board id is not permuted) and the battery caught it. The
# setup vertex is now DERIVED: it is the player's only building with no incident
# road of its own.

# Phases. PHASE_TRADE_OFFER covers BOTH plies of an announcement (ask, then offer):
# which one is pending is read off the composer's PD_TRADE_RECV, so no fourth phase.
PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD = 0, 1
PHASE_ROLL, PHASE_DISCARD, PHASE_MOVE_ROBBER, PHASE_MAIN = 2, 3, 4, 5
PHASE_ROAD_BUILDING, PHASE_TRADE_OFFER, PHASE_TRADE_ANSWER, PHASE_TRADE_ACCEPT = 6, 7, 8, 9
N_PHASES = 10

############################## ACTION LAYOUT ##################################
#
# Disjoint ranges, no output neuron ever means two different things.
# Every block is defined by an offset and a size so that nothing is ever
# hard-coded twice. See CatanLogicNumba.py header for the encoding of each.

A_ROAD = 0                                      # 72 : edge id
A_SETTLEMENT = A_ROAD + N_EDGES                 # 54 : vertex id
A_CITY = A_SETTLEMENT + N_VERTICES              # 54 : vertex id
A_BUY_DEV = A_CITY + N_VERTICES                 # 1
A_PLAY_DEV = A_BUY_DEV + 1                      # 2 : knight / road building. Monopoly and
                                                #     Year of Plenty ARE their parameter
                                                #     blocks below: one action, one neuron
A_ROBBER = A_PLAY_DEV + 2                       # 19*P : hex * victim, victim 0 = nobody
A_ROLL = A_ROBBER + N_HEXES * N_PLAYERS         # 1 : roll the dice
A_MONOPOLY = A_ROLL + 1                         # 5 : resource id
A_YEAR_OF_PLENTY = A_MONOPOLY + N_RESOURCES     # 15 : unordered pair of resources
A_BANK_TRADE = A_YEAR_OF_PLENTY + 15            # 20 : give*4 + get_shifted
A_DISCARD = A_BANK_TRADE + 20                   # 5 : resource id, one card at a time
A_END_TURN = A_DISCARD + N_RESOURCES            # 1
N_ACTIONS_V1 = A_END_TURN + 1

# ---- Player-to-player trade -------------------------------------------------
#
# An announcement is FACTORISED over two plies instead of enumerated. Every
# multiset of 1..3 cards over the 5 resources (5 + 15 + 35 = 55) can be asked for
# and offered, so a flat id per (ask, offer) PAIR would need 55*55 = 3025; the
# split needs 55 + 55 and loses nothing. RECV comes first because what a player
# needs follows from what it is building, while what it will part with depends on
# what it is asking for.
TRADE_SETS = np.array([
	[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],
	[2,0,0,0,0], [1,1,0,0,0], [1,0,1,0,0], [1,0,0,1,0], [1,0,0,0,1],
	[0,2,0,0,0], [0,1,1,0,0], [0,1,0,1,0], [0,1,0,0,1], [0,0,2,0,0],
	[0,0,1,1,0], [0,0,1,0,1], [0,0,0,2,0], [0,0,0,1,1], [0,0,0,0,2],
	[3,0,0,0,0], [2,1,0,0,0], [2,0,1,0,0], [2,0,0,1,0], [2,0,0,0,1],
	[1,2,0,0,0], [1,1,1,0,0], [1,1,0,1,0], [1,1,0,0,1], [1,0,2,0,0],
	[1,0,1,1,0], [1,0,1,0,1], [1,0,0,2,0], [1,0,0,1,1], [1,0,0,0,2],
	[0,3,0,0,0], [0,2,1,0,0], [0,2,0,1,0], [0,2,0,0,1], [0,1,2,0,0],
	[0,1,1,1,0], [0,1,1,0,1], [0,1,0,2,0], [0,1,0,1,1], [0,1,0,0,2],
	[0,0,3,0,0], [0,0,2,1,0], [0,0,2,0,1], [0,0,1,2,0], [0,0,1,1,1],
	[0,0,1,0,2], [0,0,0,3,0], [0,0,0,2,1], [0,0,0,1,2], [0,0,0,0,3],
], dtype=np.int8)
N_TRADE_SETS = TRADE_SETS.shape[0]              # 55

A_TRADE_RECV = N_ACTIONS_V1                     # 55 : the multiset I ask FOR
A_TRADE_GIVE = A_TRADE_RECV + N_TRADE_SETS      # 55 : the multiset I offer in exchange
A_TRADE_OK = A_TRADE_GIVE + N_TRADE_SETS        # 1 : a responder accepts the turn player's offer
A_TRADE_NO = A_TRADE_OK + 1                     # 1 : a responder declines without countering
A_TRADE_ACCEPT = A_TRADE_NO + 1                 # P : the turn player picks a counter-offer.
                                                #     Relative player id, 0 = refuse them all
                                                #     (same convention as A_ROBBER's victim).
N_ACTIONS = A_TRADE_ACCEPT + N_PLAYERS

# Unordered pairs of resources, for Year of Plenty (id -> the two resource ids)
YOP_PAIRS = np.array([[a, b] for a in range(N_RESOURCES) for b in range(a, N_RESOURCES)], dtype=np.int8)

# Image of every action id under each isometry. int16: N_ACTIONS exceeds 127.
# Starting from the identity is what leaves the trade block alone: its ids name
# RESOURCES, which no isometry permutes. CatanConstantsTest asserts that, rather
# than leaving it to this initialisation being read correctly.
ISO_ACTION = np.tile(np.arange(N_ACTIONS, dtype=np.int16), (N_ISOMETRIES, 1))
for _s in range(N_ISOMETRIES):
	for _e in range(N_EDGES):
		ISO_ACTION[_s, A_ROAD + _e] = A_ROAD + int(ISO_EDGE[_s, _e])
	for _v in range(N_VERTICES):
		ISO_ACTION[_s, A_SETTLEMENT + _v] = A_SETTLEMENT + int(ISO_VERTEX[_s, _v])
		ISO_ACTION[_s, A_CITY + _v] = A_CITY + int(ISO_VERTEX[_s, _v])
	for _h in range(N_HEXES):
		for _t in range(N_PLAYERS):
			ISO_ACTION[_s, A_ROBBER + _h * N_PLAYERS + _t] = A_ROBBER + int(ISO_HEX[_s, _h]) * N_PLAYERS + _t
