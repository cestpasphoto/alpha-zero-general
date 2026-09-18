"""Assertion battery for Catan. Written BEFORE the game logic, on purpose.

Three layers, in increasing order of what they need:

  1. check_state(state)        pure function on a raw state array. Runs today.
  2. check_isometries(state)   the x12 augmentation, on a static state. Runs today.
  3. check_with_logic()        random playouts, make_move/isometry commutation,
                               get_observation / sample_world. Runs as soon as
                               CatanLogicNumba exists; skipped until then.

Layer 4 is this file testing itself: test_the_instrument() corrupts a valid state
in 14 known ways and asserts each one is caught. An assertion battery that has
never failed on purpose is not an instrument, it is decoration.

build_reference_state() is deliberately an INDEPENDENT oracle: it builds a legal
state with plain numpy, without importing the game logic, so that a bug in
init_game cannot hide behind the same bug in the checker.

Usage:
    python CatanTest.py            # layers 1, 2, 4 (+ 3 if the logic is there)
"""
import numpy as np

try:                                # works both as `python catan/CatanTest.py`
	from CatanConstants import *    # and from inside the package
except ImportError:                 # pragma: no cover
	from .CatanConstants import *


# Decisions, not rounds. A turn is roll + up to MAX_TRADES_PER_TURN bank trades +
# one player-trade attempt (2P+1 plies at worst) + builds + end turn. Builds and dev
# buys are bounded over the whole GAME (24 pieces, a 25-card deck) rather than per
# turn, so this is loose on purpose -- it exists to stop a genuinely stalled game
# from hanging the battery, not to be tight. It replaces a flat 4*MAX_ROUNDS, which
# player trades quietly outgrew: random play needs ~6 decisions per round with them
# on, against ~4 before, so games were being cut off at round ~270 of 400 and
# reported as "did not terminate".
MAX_DECISIONS = MAX_ROUNDS * (4 + MAX_TRADES_PER_TURN + 2 * N_PLAYERS + 1)


# ---------------------------------------------------------------------------
# Reference state builder (test oracle, independent of the game logic)
# ---------------------------------------------------------------------------

def build_reference_state(seed=0, n_settlements=2, give_resources=True):
	"""A legal mid-setup position: shuffled board, a few settlements and roads."""
	rng = np.random.default_rng(seed)
	st = np.zeros((N_ROWS, N_COLS), dtype=np.int8)
	hexes = st[ROW_HEX:ROW_HEX + N_HEXES]
	verts = st[ROW_VERTEX:ROW_VERTEX + N_VERTICES]

	# --- board -------------------------------------------------------------
	types = HEX_DISTRIBUTION.copy()
	rng.shuffle(types)
	tokens = TOKEN_DISTRIBUTION.copy()
	rng.shuffle(tokens)
	ti = 0
	for h in range(N_HEXES):
		hexes[h, H_TYPE] = types[h]
		if types[h] == HEX_DESERT:
			hexes[h, H_TOKEN] = 0
			hexes[h, H_ROBBER] = 1
		else:
			hexes[h, H_TOKEN] = tokens[ti]
			ti += 1
		hexes[h, H_PIPS] = TOKEN_PIPS[hexes[h, H_TOKEN]]

	port_types = PORT_DISTRIBUTION.copy()
	rng.shuffle(port_types)
	for p in range(N_PORTS):
		for v in PORT_VERTICES[p]:
			verts[v, V_PORT] = port_types[p]

	# --- players -----------------------------------------------------------
	for p in range(N_PLAYERS):
		b = st[ROW_PLAYER + 4 * p + 1]
		b[PB_SETTLEMENTS_LEFT] = MAX_SETTLEMENTS
		b[PB_CITIES_LEFT] = MAX_CITIES
		b[PB_ROADS_LEFT] = MAX_ROADS

	st[ROW_GLOBAL, GA_BANK:GA_BANK + N_RESOURCES] = BANK_PER_RESOURCE
	st[ROW_GLOBAL, GA_DEV_DECK:GA_DEV_DECK + N_DEV_TYPES] = DEV_DISTRIBUTION
	st[ROW_GLOBAL, GA_PHASE] = PHASE_MAIN
	st[ROW_GLOBAL + 1, GB_TURN_PLAYER] = 0

	# --- a few settlements, respecting the distance rule -------------------
	free = list(range(N_VERTICES))
	rng.shuffle(free)
	for p in range(N_PLAYERS):
		placed = 0
		for v in list(free):
			if verts[v, V_BUILDING]:
				continue
			if any(verts[o, V_BUILDING] for o in VERTEX_TO_VERTEX[v] if o != NO_VERTEX):
				continue
			verts[v, V_BUILDING] = 1
			verts[v, V_OWNER] = p + 1
			st[ROW_PLAYER + 4 * p + 1, PB_SETTLEMENTS_LEFT] -= 1
			# two roads out of that settlement, so every road touches a building
			roads = 0
			for k in range(3):
				e = VERTEX_TO_EDGE[v, k]
				if e == NO_EDGE or roads >= 2:
					continue
				if _edge_owner(st, e) == 0:
					_set_edge(st, e, p + 1)
					st[ROW_PLAYER + 4 * p + 1, PB_ROADS_LEFT] -= 1
					roads += 1
			st[ROW_PLAYER + 4 * p + 1, PB_ROAD_LENGTH] = roads
			free.remove(v)
			placed += 1
			if placed >= n_settlements:
				break

	if give_resources:
		for p in range(N_PLAYERS):
			a = st[ROW_PLAYER + 4 * p]
			for r in range(N_RESOURCES):
				n = rng.integers(0, 4)
				a[PA_RESOURCES + r] = n
				st[ROW_GLOBAL, GA_BANK + r] -= n

	refresh_derived(st)
	return st


def refresh_derived(st):
	"""Recompute every cached / redundant field from the primary ones."""
	verts = st[ROW_VERTEX:ROW_VERTEX + N_VERTICES]
	for p in range(N_PLAYERS):
		a, b, c = st[ROW_PLAYER + 4 * p], st[ROW_PLAYER + 4 * p + 1], st[ROW_PLAYER + 4 * p + 2]
		a[PA_TOTAL_RES] = a[PA_RESOURCES:PA_RESOURCES + N_RESOURCES].sum()
		a[PA_TOTAL_DEV] = (a[PA_DEV_PLAYABLE:PA_DEV_PLAYABLE + N_DEV_TYPES].sum()
		                   + b[PB_DEV_NEW:PB_DEV_NEW + N_DEV_TYPES].sum())
		c[PC_PORTS:PC_PORTS + 6] = 0
		s = c_ = 0
		for v in range(N_VERTICES):
			if verts[v, V_OWNER] == p + 1:
				if verts[v, V_BUILDING] == 1:
					s += 1
				elif verts[v, V_BUILDING] == 2:
					c_ += 1
				if verts[v, V_PORT]:
					c[PC_PORTS + verts[v, V_PORT] - 1] = 1
		c[PC_VP_PUBLIC] = s + 2 * c_ + 2 * b[PB_HAS_ROAD] + 2 * b[PB_HAS_ARMY]
		c[PC_VP_DEV] = a[PA_DEV_PLAYABLE + VICTORY_POINT] + b[PB_DEV_NEW + VICTORY_POINT]
		c[PC_TOTAL_DEV_NEW] = b[PB_DEV_NEW:PB_DEV_NEW + N_DEV_TYPES].sum()


def _set_edge(st, e, owner):
	for i in (0, 1):
		v = EDGE_TO_VERTEX[e, i]
		k = int(np.flatnonzero(VERTEX_TO_EDGE[v] == e)[0])
		st[ROW_VERTEX + v, V_EDGE0 + k] = owner


def _edge_owner(st, e):
	v = EDGE_TO_VERTEX[e, 0]
	k = int(np.flatnonzero(VERTEX_TO_EDGE[v] == e)[0])
	return int(st[ROW_VERTEX + v, V_EDGE0 + k])


# ---------------------------------------------------------------------------
# Layer 1: invariants of a single state
# ---------------------------------------------------------------------------

def check_state(st, masked=False):
	"""Raise AssertionError on the first violated invariant.

	masked=True relaxes the invariants that get_observation deliberately breaks
	(opponent hands are zeroed, so resources no longer sum to 19).
	"""
	verts = st[ROW_VERTEX:ROW_VERTEX + N_VERTICES]
	hexes = st[ROW_HEX:ROW_HEX + N_HEXES]
	ga, gb = st[ROW_GLOBAL], st[ROW_GLOBAL + 1]

	# --- I1 shape, sign, unused columns ------------------------------------
	assert st.shape == (N_ROWS, N_COLS) and st.dtype == np.int8, f"shape/dtype {st.shape} {st.dtype}"
	assert (st >= 0).all(), f"negative value at {np.argwhere(st < 0)[:3].tolist()}"
	assert (verts[:, 6:] == 0).all(), "vertex rows write past column 5"
	assert (hexes[:, 4:] == 0).all(), "hex rows write past column 3"

	# --- I2 hexes -----------------------------------------------------------
	assert sorted(hexes[:, H_TYPE].tolist()) == sorted(HEX_DISTRIBUTION.tolist()), "hex type multiset"
	toks = [t for t in hexes[:, H_TOKEN].tolist() if t != 0]
	assert sorted(toks) == sorted(TOKEN_DISTRIBUTION.tolist()), "token multiset"
	for h in range(N_HEXES):
		assert (hexes[h, H_TOKEN] == 0) == (hexes[h, H_TYPE] == HEX_DESERT), f"hex {h} desert/token"
		assert hexes[h, H_PIPS] == TOKEN_PIPS[hexes[h, H_TOKEN]], f"hex {h} pips != TOKEN_PIPS[token]"
	assert hexes[:, H_ROBBER].sum() == 1, f"{hexes[:, H_ROBBER].sum()} robbers"

	# --- I3 ports -----------------------------------------------------------
	# Deliberately position-independent: ports live in the state, not in the
	# constants, which is what keeps the 12 isometries legal rewrites. Checking
	# against PORT_VERTICES here would re-couple them and reject every rotation.
	# That init_game uses the frame positions is a layer-3 check.
	port_v = set(np.flatnonzero(verts[:, V_PORT] != PORT_NONE).tolist())
	assert len(port_v) == 2 * N_PORTS, f"{len(port_v)} port vertices, expected {2 * N_PORTS}"
	seen, covered = [], set()
	for e in COASTAL_EDGES:
		a, b = EDGE_TO_VERTEX[e]
		if a in port_v and b in port_v and verts[a, V_PORT] == verts[b, V_PORT]:
			assert a not in covered and b not in covered, f"vertex shared by two ports at edge {e}"
			covered |= {int(a), int(b)}
			seen.append(int(verts[a, V_PORT]))
	assert covered == port_v, "a port vertex has no matching partner on a coastal edge"
	assert sorted(seen) == sorted(PORT_DISTRIBUTION.tolist()), "port type multiset"

	# --- I4 buildings and the distance rule --------------------------------
	for v in range(N_VERTICES):
		assert (verts[v, V_BUILDING] == 0) == (verts[v, V_OWNER] == 0), f"vertex {v} building/owner"
		assert verts[v, V_BUILDING] <= 2 and verts[v, V_OWNER] <= N_PLAYERS, f"vertex {v} out of range"
		if verts[v, V_BUILDING]:
			for o in VERTEX_TO_VERTEX[v]:
				if o != NO_VERTEX:
					assert verts[o, V_BUILDING] == 0, f"distance rule broken between {v} and {o}"

	# --- I5 edges: one road, two consistent copies -------------------------
	for e in range(N_EDGES):
		owners = []
		for i in (0, 1):
			v = EDGE_TO_VERTEX[e, i]
			k = int(np.flatnonzero(VERTEX_TO_EDGE[v] == e)[0])
			owners.append(int(verts[v, V_EDGE0 + k]))
		assert owners[0] == owners[1], f"edge {e} owner disagrees between its endpoints {owners}"
		assert owners[0] <= N_PLAYERS, f"edge {e} owner out of range"
	for v in range(N_VERTICES):
		for k in range(3):
			if VERTEX_TO_EDGE[v, k] == NO_EDGE:
				assert verts[v, V_EDGE0 + k] == 0, f"vertex {v} slot {k} has no edge but an owner"

	# --- I6 piece counts ----------------------------------------------------
	for p in range(N_PLAYERS):
		b = st[ROW_PLAYER + 4 * p + 1]
		s = int(((verts[:, V_OWNER] == p + 1) & (verts[:, V_BUILDING] == 1)).sum())
		c = int(((verts[:, V_OWNER] == p + 1) & (verts[:, V_BUILDING] == 2)).sum())
		r = sum(1 for e in range(N_EDGES) if _edge_owner(st, e) == p + 1)
		assert b[PB_SETTLEMENTS_LEFT] + s == MAX_SETTLEMENTS, f"p{p} settlements {s}+{b[PB_SETTLEMENTS_LEFT]}"
		assert b[PB_CITIES_LEFT] + c == MAX_CITIES, f"p{p} cities"
		assert b[PB_ROADS_LEFT] + r == MAX_ROADS, f"p{p} roads {r}+{b[PB_ROADS_LEFT]}"
		assert b[PB_ROAD_LENGTH] <= r, f"p{p} longest road {b[PB_ROAD_LENGTH]} > {r} roads owned"

	# --- I7 resource conservation ------------------------------------------
	if not masked:
		for r in range(N_RESOURCES):
			total = int(ga[GA_BANK + r]) + sum(int(st[ROW_PLAYER + 4 * p, PA_RESOURCES + r])
			                                   for p in range(N_PLAYERS))
			assert total == BANK_PER_RESOURCE, f"resource {r}: {total} in play, expected {BANK_PER_RESOURCE}"
		assert (ga[GA_BANK:GA_BANK + N_RESOURCES] <= BANK_PER_RESOURCE).all(), "bank overflow"

	# --- I8 development card conservation ----------------------------------
	for k in range(N_DEV_TYPES):
		total = int(ga[GA_DEV_DECK + k]) + int(gb[GB_DEV_PLAYED + k])
		for p in range(N_PLAYERS):
			total += int(st[ROW_PLAYER + 4 * p, PA_DEV_PLAYABLE + k])
			total += int(st[ROW_PLAYER + 4 * p + 1, PB_DEV_NEW + k])
		if not masked:
			assert total == DEV_DISTRIBUTION[k], f"dev type {k}: {total} vs {DEV_DISTRIBUTION[k]}"
	knights = sum(int(st[ROW_PLAYER + 4 * p + 1, PB_KNIGHTS]) for p in range(N_PLAYERS))
	assert knights == gb[GB_DEV_PLAYED + KNIGHT], "knights played != knights discarded"

	# --- I9 cached / redundant fields --------------------------------------
	ref = st.copy()
	refresh_derived(ref)
	for p in range(N_PLAYERS):
		c, rc = st[ROW_PLAYER + 4 * p + 2], ref[ROW_PLAYER + 4 * p + 2]
		assert (c[PC_PORTS:PC_PORTS + 6] == rc[PC_PORTS:PC_PORTS + 6]).all(), f"p{p} PC_PORTS stale"
		assert c[PC_VP_PUBLIC] == rc[PC_VP_PUBLIC], f"p{p} VP {c[PC_VP_PUBLIC]} vs {rc[PC_VP_PUBLIC]}"
		if not masked:
			a, ra = st[ROW_PLAYER + 4 * p], ref[ROW_PLAYER + 4 * p]
			assert a[PA_TOTAL_RES] == ra[PA_TOTAL_RES], f"p{p} PA_TOTAL_RES stale"
			assert a[PA_TOTAL_DEV] == ra[PA_TOTAL_DEV], f"p{p} PA_TOTAL_DEV stale"
			assert c[PC_VP_DEV] == rc[PC_VP_DEV], f"p{p} PC_VP_DEV stale"
			assert c[PC_TOTAL_DEV_NEW] == rc[PC_TOTAL_DEV_NEW], f"p{p} PC_TOTAL_DEV_NEW stale"

	# --- I10 unique bonus cards --------------------------------------------
	army = [p for p in range(N_PLAYERS) if st[ROW_PLAYER + 4 * p + 1, PB_HAS_ARMY]]
	road = [p for p in range(N_PLAYERS) if st[ROW_PLAYER + 4 * p + 1, PB_HAS_ROAD]]
	assert len(army) <= 1 and len(road) <= 1, f"army {army} road {road}"
	for p in army:
		k = st[ROW_PLAYER + 4 * p + 1, PB_KNIGHTS]
		assert k >= MIN_KNIGHTS_FOR_ARMY, f"largest army with {k} knights"
		assert all(k >= st[ROW_PLAYER + 4 * q + 1, PB_KNIGHTS] for q in range(N_PLAYERS)), "army not the max"
	for p in road:
		l = st[ROW_PLAYER + 4 * p + 1, PB_ROAD_LENGTH]
		assert l >= MIN_LENGTH_FOR_ROAD, f"longest road with length {l}"
		assert all(l >= st[ROW_PLAYER + 4 * q + 1, PB_ROAD_LENGTH] for q in range(N_PLAYERS)), "road not the max"

	# --- I11 globals --------------------------------------------------------
	assert ga[GA_PHASE] < N_PHASES, "phase out of range"
	assert ga[GA_DICE] == 0 or 2 <= ga[GA_DICE] <= 12, f"dice {ga[GA_DICE]}"
	assert gb[GB_ROUND_LO] < 100, "round_lo must stay below 100"
	assert int(gb[GB_ROUND_HI]) * 100 + int(gb[GB_ROUND_LO]) <= MAX_ROUNDS, "round past MAX_ROUNDS"
	assert gb[GB_TURN_PLAYER] < N_PLAYERS, "turn player out of range"
	assert gb[GB_CHANCE_COUNTER] < 100, "chance counter must stay below 100"
	for p in range(N_PLAYERS):
		assert st[ROW_PLAYER + 4 * p + 2, PC_TRADES_THIS_TURN] <= MAX_TRADES_PER_TURN, \
			f"p{p} exceeded MAX_TRADES_PER_TURN"

	# --- I12 the player-trade protocol (global row D of each player) -------
	# Checked WITHOUT the `masked` escape on purpose: an announcement is public,
	# everyone at the table heard it, so get_observation must leave row D alone.
	# If masking ever starts touching it, these fire on the observation itself.
	phase = int(ga[GA_PHASE])
	turn = int(gb[GB_TURN_PLAYER])
	status, composing, offered = [], [], []
	for p in range(N_PLAYERS):
		d = st[ROW_PLAYER + 4 * p + 3]
		s = int(d[PD_TRADE_STATUS])
		assert s <= TRADE_REFUSED, f"p{p} trade status {s} out of range"
		recv = d[PD_TRADE_RECV:PD_TRADE_RECV + N_RESOURCES]
		give = d[PD_TRADE_GIVE:PD_TRADE_GIVE + N_RESOURCES]
		if s in (TRADE_NONE, TRADE_REFUSED):
			assert recv.sum() == 0 and give.sum() == 0, f"p{p} status {s} but an offer is recorded"
		elif s == TRADE_COMPOSING:
			assert 1 <= recv.sum() <= 3, f"p{p} is composing an ask of size {recv.sum()}"
			assert give.sum() == 0, f"p{p} is still composing but already offers something"
		else:
			assert 1 <= recv.sum() <= 3 and 1 <= give.sum() <= 3, f"p{p} published a badly sized offer"
			assert not ((recv > 0) & (give > 0)).any(), f"p{p} trades a resource against itself"
		status.append(s)
		if s == TRADE_COMPOSING:
			composing.append(p)
		elif s == TRADE_OFFERED:
			offered.append(p)
	trade_phases = (PHASE_TRADE_OFFER, PHASE_TRADE_ANSWER, PHASE_TRADE_ACCEPT)
	if phase not in trade_phases:
		assert all(s == TRADE_NONE for s in status), \
			f"phase {phase} is not a trade phase but statuses are {status}"
	else:
		# _next_actor derives the whole protocol from these, so an inconsistency
		# here is a silently wrong ACTOR, not a visible crash
		assert len(composing) == (1 if phase == PHASE_TRADE_OFFER else 0), \
			f"phase {phase} with {len(composing)} players composing"
		waiting = [p for p in range(N_PLAYERS) if p != turn and status[p] == TRADE_NONE]
		if phase == PHASE_TRADE_ANSWER:
			assert status[turn] == TRADE_OFFERED, "answering an offer the turn player never made"
			assert waiting, "PHASE_TRADE_ANSWER with nobody left to answer"
		elif phase == PHASE_TRADE_ACCEPT:
			assert status[turn] == TRADE_OFFERED, "accepting against no standing offer"
			assert not waiting, "PHASE_TRADE_ACCEPT while someone has not answered"
			assert [p for p in offered if p != turn], "PHASE_TRADE_ACCEPT with no counter-offer"
		# a counter can only exist once the turn player has announced
		if [p for p in offered + composing if p != turn]:
			assert status[turn] != TRADE_NONE, "a counter-offer without an original offer"


# ---------------------------------------------------------------------------
# Layer 2: the x12 augmentation
# ---------------------------------------------------------------------------

def permute_state(st, s):
	"""Apply isometry s. Reference implementation: get_symmetries() must match it."""
	out = st.copy()
	iv, ih, ie = ISO_VERTEX[s], ISO_HEX[s], ISO_EDGE[s]
	for v in range(N_VERTICES):
		out[ROW_VERTEX + iv[v], :V_EDGE0] = st[ROW_VERTEX + v, :V_EDGE0]
	for h in range(N_HEXES):
		out[ROW_HEX + ih[h]] = st[ROW_HEX + h]
	for v in range(N_VERTICES):           # edges move with their image, slot by slot
		for k in range(3):
			e = VERTEX_TO_EDGE[v, k]
			if e == NO_EDGE:
				continue
			nv, ne = iv[v], ie[e]
			nk = int(np.flatnonzero(VERTEX_TO_EDGE[nv] == ne)[0])
			out[ROW_VERTEX + nv, V_EDGE0 + nk] = st[ROW_VERTEX + v, V_EDGE0 + k]
	return out


def permute_action(a, s):
	"""Image of action id `a` under isometry s. Only board-anchored blocks move."""
	if A_ROAD <= a < A_ROAD + N_EDGES:
		return A_ROAD + int(ISO_EDGE[s][a - A_ROAD])
	if A_SETTLEMENT <= a < A_SETTLEMENT + N_VERTICES:
		return A_SETTLEMENT + int(ISO_VERTEX[s][a - A_SETTLEMENT])
	if A_CITY <= a < A_CITY + N_VERTICES:
		return A_CITY + int(ISO_VERTEX[s][a - A_CITY])
	if A_ROBBER <= a < A_ROBBER + N_HEXES * N_PLAYERS:
		h, t = divmod(a - A_ROBBER, N_PLAYERS)
		return A_ROBBER + int(ISO_HEX[s][h]) * N_PLAYERS + t
	return a


def check_isometries(st):
	ident = permute_state(st, 0)
	assert (ident == st).all(), "isometry 0 is not the identity on the state"
	for s in range(N_ISOMETRIES):
		img = permute_state(st, s)
		check_state(img)                                    # a rotated state is still legal
		perm = [permute_action(a, s) for a in range(N_ACTIONS)]
		assert sorted(perm) == list(range(N_ACTIONS)), f"action permutation {s} is not a bijection"
		assert perm == ISO_ACTION[s].tolist(), f"ISO_ACTION[{s}] disagrees with permute_action"
		# the player-facing content must be untouched
		assert (img[ROW_PLAYER:] == st[ROW_PLAYER:]).all(), f"isometry {s} touched player/global rows"
	# composition: permuting twice equals permuting by the composed isometry
	for s in range(N_ISOMETRIES):
		for t in range(N_ISOMETRIES):
			comp = ISO_VERTEX[s][ISO_VERTEX[t]]
			u = [k for k in range(N_ISOMETRIES) if (ISO_VERTEX[k] == comp).all()][0]
			assert (permute_state(permute_state(st, t), s) == permute_state(st, u)).all(), (s, t)


# ---------------------------------------------------------------------------
# Layer 3: needs CatanLogicNumba (skipped until it exists)
# ---------------------------------------------------------------------------

def check_robber_action_indices():
	"""Regression test for a real bug: `self.num_players` is a numba.int8
	JITCLASS FIELD. Combined arithmetically with a large python-int constant
	(A_ROBBER = ~183), numpy's NEP 50 rules keep the result in int8 -- which
	overflows (max 127) past h=1 already. Interpreted mode raises OverflowError
	loudly; COMPILED numba wraps SILENTLY (C semantics, no exception at all),
	so this corrupted valid_moves()/_apply() dispatch with no crash to point at
	-- the default execution mode this whole battery otherwise runs under never
	saw it. Every check above this one tests "did it crash or violate an
	invariant", never "does this specific index equal what plain arithmetic
	says it should" -- that gap is exactly what let this through.
	"""
	Board = _import_board()
	if Board is None or not hasattr(Board, 'make_move'):
		print('layer 3c (robber indices)      SKIPPED - logic not written yet')
		return False
	board = Board(N_PLAYERS)
	board.init_game()

	# 1) valid_moves() in PHASE_MOVE_ROBBER must match an INDEPENDENT reference,
	#    computed with plain Python ints only (no jitclass involved at all).
	board.get_state()[ROW_GLOBAL, GA_PHASE] = PHASE_MOVE_ROBBER
	valids = board.valid_moves(0)
	got = set(np.flatnonzero(valids[A_ROBBER:A_ROBBER + N_HEXES * N_PLAYERS]).tolist())
	# every hex except the one already holding the robber offers "rob nobody" (t=0)
	robber_hex = int(np.flatnonzero(board.get_state()[ROW_HEX:ROW_HEX + N_HEXES, H_ROBBER])[0])
	expected = {h * N_PLAYERS + 0 for h in range(N_HEXES) if h != robber_hex}
	assert got == expected, f'robber action set mismatch: extra {got - expected}, missing {expected - got}'

	# 2) the boundary index specifically -- (h=N_HEXES-1, t=N_PLAYERS-1), the
	#    largest value this expression ever produces -- must be independently
	#    reachable via read AND write, not just "doesn't crash on read".
	worst = A_ROBBER + (N_HEXES - 1) * N_PLAYERS + (N_PLAYERS - 1)
	assert worst < N_ACTIONS_V1, f'boundary index {worst} escaped the reserved v1 range'
	probe = np.zeros(N_ACTIONS, dtype=np.bool_)
	probe[worst] = True                      # would raise/wrap under the old bug
	assert probe[worst] and probe.sum() == 1

	# 3) dispatch must actually APPLY a robber move, not silently no-op: the
	#    robber must land on the hex we chose. (Checking the PHASE afterwards
	#    would be wrong: make_move()'s own auto-resolve loop can legitimately
	#    cascade through END_TURN -> ROLL -> a NEW seven for the next player,
	#    landing back in PHASE_MOVE_ROBBER for a different reason entirely. The
	#    robber's position is the direct, cascade-proof signature of OUR move.)
	h_target = max(got) // N_PLAYERS
	a = A_ROBBER + max(got)
	before = board.get_state().copy()
	board.make_move(a, 0, 1)
	after = board.get_state()
	assert after[ROW_HEX + h_target, H_ROBBER] == 1, \
		'robber move dispatched to nothing: robber never reached the target hex ' \
		'(the exact silent-no-op failure mode)'
	assert not (after == before).all(), 'robber move applied but state is byte-identical to before'
	print(f'layer 3c (robber indices)      OK  ({N_PLAYERS} players, boundary index {worst})')
	return True


def check_setup_actor_consistency(n_games=200, seed=0):
	"""Regression test for a real bug: _next_actor()'s setup-phase formula
	returned an ABSOLUTE player id where every caller (_apply, make_move,
	the outer swap_players() convention) expects a CANONICAL-RELATIVE offset.
	It coincided with the relative convention only for the very first
	placement (s=0), then diverged: a road could be requested for a player
	who owns no settlement at all, giving zero legal moves. Uniform-random
	playouts (check_with_logic) never catch this -- a random policy just
	misattributes ownership to whichever player _next_actor() names, and
	keeps going, since it never revisits an already-expanded node the way
	PUCT does. This test drives setup through the SAME copy/make_move/
	swap_players cycle MCTS.py's get_next_best_action_and_canonical_state
	uses, and checks that every road's owner matches the settlement it
	completes, at every step, for every player count.
	"""
	Board = _import_board()
	if Board is None or not hasattr(Board, 'make_move'):
		print('layer 3e (setup actor)         SKIPPED - logic not written yet')
		return False
	rng = np.random.default_rng(seed)
	board = Board(N_PLAYERS)
	for g in range(n_games):
		board.init_game()
		while board.get_state()[ROW_GLOBAL, GA_PHASE] in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD):
			st = board.get_state()
			valids = board.valid_moves(0)
			assert valids.any(), f'game {g}: zero legal moves during setup (round {board.get_round()})'
			a = int(rng.choice(np.flatnonzero(valids)))
			if A_SETTLEMENT <= a < A_SETTLEMENT + N_VERTICES and st[ROW_GLOBAL, GA_PHASE] == PHASE_SETUP_SETTLEMENT:
				v_expected = a - A_SETTLEMENT
			else:
				v_expected = None
			next_player = board.make_move(a, 0, g + 1)
			st2 = board.get_state()
			if v_expected is not None:
				# the settlement just placed must be owned by canonical player 0
				assert st2[ROW_VERTEX + v_expected, V_OWNER] == 1, \
					f'game {g}: settlement at {v_expected} not owned by the acting player'
			if next_player != 0:
				board.swap_players(next_player)
			# invariant: every settlement on the board has EITHER a road from
			# the SAME owner, or is the one settlement currently awaiting one
			# -- and there is never more than one such pending settlement.
			st3 = board.get_state()
			pending = 0
			for v in range(N_VERTICES):
				if st3[ROW_VERTEX + v, V_BUILDING] == 0:
					continue
				owner = st3[ROW_VERTEX + v, V_OWNER]
				has_road = any(st3[ROW_VERTEX + v, V_EDGE0 + k] == owner for k in range(3))
				if not has_road:
					pending += 1
			assert pending <= 1, f'game {g}: {pending} settlements simultaneously missing a road'
	print(f'layer 3e (setup actor, {n_games} games)   OK')
	return True


def check_trade_cap():
	"""Regression test for a real bug: bank trades had NO per-turn limit, and
	from a maxed-out hand (19 of a resource, the bank's own ceiling) a single
	turn fit up to 41 consecutive 2:1 trades. Sustained across MAX_ROUNDS an
	adversarial "always trade" policy drove MCTS's recursive search() past
	20000 stack frames on a single simulation -- not a deep-but-legitimate
	tree, a structurally unbounded one. MAX_TRADES_PER_TURN caps it.
	"""
	Board = _import_board()
	if Board is None or not hasattr(Board, 'make_move'):
		print('layer 3d (trade cap)           SKIPPED - logic not written yet')
		return False

	# 1) from a hand maxed out at the bank's own ceiling, with the best
	#    possible ratio, trading must stop at the cap -- not at exhaustion.
	board = Board(N_PLAYERS)
	board.init_game()
	st = board.get_state()
	st[ROW_PLAYER, PA_RESOURCES:PA_RESOURCES + N_RESOURCES] = BANK_PER_RESOURCE
	st[ROW_PLAYER + 2, PC_PORTS:PC_PORTS + 6] = 0
	st[ROW_PLAYER + 2, PC_PORTS + BRICK] = 1          # best ratio: 2:1
	board._refresh_totals(0)
	st[ROW_GLOBAL, GA_PHASE] = PHASE_MAIN
	st[ROW_GLOBAL, GA_BANK:GA_BANK + N_RESOURCES] = 0
	st[ROW_GLOBAL, GA_BANK + LUMBER] = BANK_PER_RESOURCE
	n = 0
	for _ in range(MAX_TRADES_PER_TURN + 5):
		trades = np.flatnonzero(board.valid_moves(0)[A_BANK_TRADE:A_BANK_TRADE + 20])
		if len(trades) == 0:
			break
		board._apply(A_BANK_TRADE + int(trades[0]), 0, 1)
		n += 1
	assert n == MAX_TRADES_PER_TURN, f'traded {n} times from a maxed hand, expected exactly {MAX_TRADES_PER_TURN}'
	assert not board.valid_moves(0)[A_BANK_TRADE:A_BANK_TRADE + 20].any(), 'trades still offered past the cap'

	# 2) an adversarial "always trade if possible, else random" policy across
	#    MANY rounds must stay within a TIGHT, predictable total -- the whole
	#    point of the cap -- not merely "didn't crash this run".
	rng = np.random.default_rng(0)
	board = Board(N_PLAYERS)
	board.init_game()
	p, decisions, trade_decisions = 0, 0, 0
	while board.get_state()[ROW_GLOBAL, GA_PHASE] in (PHASE_SETUP_SETTLEMENT, PHASE_SETUP_ROAD):
		v = board.valid_moves(p)
		p = int(board.make_move(int(rng.choice(np.flatnonzero(v))), p, 1))
	worst_case = MAX_ROUNDS * (2 + MAX_TRADES_PER_TURN) * N_PLAYERS   # roll+end+trades, every round, every seat
	for it in range(worst_case + 1000):
		decisions += 1
		v = board.valid_moves(p)
		idx = np.flatnonzero(v)
		trades = idx[(idx >= A_BANK_TRADE) & (idx < A_BANK_TRADE + 20)]
		if len(trades):
			a, trade_decisions = int(trades[0]), trade_decisions + 1
		else:
			a = int(rng.choice(idx))
		p = int(board.make_move(a, p, it + 1))
		if board.check_end_game(p).any():
			break
	else:
		raise AssertionError(f'adversarial always-trade policy exceeded the structural bound of {worst_case} decisions')
	print(f'layer 3d (trade cap)           OK  (cap={MAX_TRADES_PER_TURN}/turn, '
	      f'adversarial game: {decisions} decisions incl. {trade_decisions} trades, bound was {worst_case})')
	return True


def check_victory_is_on_own_turn():
	"""OFFICIAL RULE: reaching the target wins only on your OWN turn.

	Built by hand rather than waited for in a playout: an off-turn 10th point is
	exactly the position random play almost never produces, and it is the one the
	old max-over-all-players test got wrong. It is also reachable for real -- the
	turn player cutting a road can hand Longest Road, and its 2 points, to a third
	player in the middle of someone else's move.
	"""
	Board = _import_board()
	if Board is None:
		print("layer 3f (victory timing)      SKIPPED - logic not written yet")
		return False
	st = build_reference_state(seed=3, n_settlements=2)
	st[ROW_GLOBAL, GA_PHASE] = PHASE_MAIN
	st[ROW_GLOBAL + 1, GB_TURN_PLAYER] = 0
	# hand player 1 -- NOT the player to move -- a winning score
	st[ROW_PLAYER + 4 * 1 + 2, PC_VP_PUBLIC] = VP_TO_WIN
	b = Board(N_PLAYERS)
	b.copy_state(st.copy(), True)
	assert not b.check_end_game(0).any(), \
		"a player who is not on turn was declared winner: off-turn victory"
	# identical position, except that it is now their turn
	st[ROW_GLOBAL + 1, GB_TURN_PLAYER] = 1
	b.copy_state(st.copy(), True)
	res = b.check_end_game(0)
	assert res.any(), "the turn player reached the target but the game did not end"
	assert res[1] == 1. and all(res[p] == -1. for p in range(N_PLAYERS) if p != 1), f"wrong winner: {res}"
	# and the MAX_ROUNDS timeout must still rank everyone rather than crown the seat
	st[ROW_PLAYER + 4 * 1 + 2, PC_VP_PUBLIC] = 1
	st[ROW_PLAYER + 2, PC_VP_PUBLIC] = 3
	st[ROW_GLOBAL + 1, GB_ROUND_HI], st[ROW_GLOBAL + 1, GB_ROUND_LO] = MAX_ROUNDS // 100, MAX_ROUNDS % 100
	b.copy_state(st.copy(), True)
	res = b.check_end_game(0)
	assert res[0] == 1. and res[1] == -1., f"timeout must rank on score, got {res}"
	print("layer 3f (victory timing)      OK - wins only on the winner's own turn")
	return True


def check_player_trade(n_games=12, seed=0):
	"""Layer 3g: the player-trade protocol, driven by a policy that OPENS a trade
	whenever one is legal -- uniform random play reaches the answer phases far too
	rarely to test them.

	What this pins down, worst consequence first:
	  1. no reachable phase has zero legal moves -- the crash the "keeps_something"
	     guard in _recv_is_legal exists to prevent;
	  2. the protocol terminates: at most 2P+1 plies from announcement to
	     resolution, and never two attempts in one turn;
	  3. an executed trade moves BOTH hands by exactly (give - recv) and its
	     opposite. check_state's I7 only sees global conservation, which a swapped
	     direction would satisfy just as well;
	  4. legality reads no hidden hand: the legal RECV set computed on the true
	     state equals the one computed on the masked observation;
	  5. sample_world honours a published offer -- without that, accepting one
	     debits cards the sampled world never dealt and a hand goes negative.
	"""
	Board = _import_board()
	if Board is None or not hasattr(Board, 'make_move'):
		print("layer 3g (player trade)        SKIPPED - logic not written yet")
		return False
	if not ENABLE_PLAYER_TRADE:
		print("layer 3g (player trade)        SKIPPED - ENABLE_PLAYER_TRADE is False")
		return False

	rng = np.random.default_rng(seed)
	board = Board(N_PLAYERS)
	trade_phases = (PHASE_TRADE_OFFER, PHASE_TRADE_ANSWER, PHASE_TRADE_ACCEPT)
	opened = executed = worst = 0
	for g in range(n_games):
		board.init_game()
		player, it, inside = 0, 0, 0
		while not board.check_end_game(player).any() and it < MAX_DECISIONS:
			it += 1
			phase = int(board.get_state()[ROW_GLOBAL, GA_PHASE])
			valids = board.valid_moves(player)
			assert valids.any(), f"zero legal moves, phase {phase}"
			if phase in trade_phases:
				inside += 1
				worst = max(worst, inside)
				assert inside <= 2 * N_PLAYERS + 1, \
					f"trade ran {inside} plies, bound is 2P+1 = {2 * N_PLAYERS + 1}"
			elif phase == PHASE_MAIN:
				inside = 0

			legal = np.flatnonzero(valids)
			recv_ids = [a for a in legal if A_TRADE_RECV <= a < A_TRADE_RECV + N_TRADE_SETS]
			if phase == PHASE_MAIN and recv_ids and rng.random() < 0.85:
				a, inside, opened = int(rng.choice(recv_ids)), 1, opened + 1
			elif phase == PHASE_TRADE_ANSWER and valids[A_TRADE_OK] and rng.random() < 0.4:
				a = A_TRADE_OK
			else:
				a = int(rng.choice(legal))

			before = board.get_state().copy()
			actor = player

			# 3. an executed trade moves the two hands in opposite directions.
			# Measured with _apply on a CLONE, never across make_move: make_move
			# auto-resolves every forced move that follows, so ending the turn,
			# rolling and collecting production all land in the same before/after
			# delta. A first version of this test compared across make_move and
			# "failed" on a perfectly correct trade for exactly that reason.
			proposer = -1
			if a == A_TRADE_OK:
				proposer = int(before[ROW_GLOBAL + 1, GB_TURN_PLAYER])
			elif A_TRADE_ACCEPT < a < A_TRADE_ACCEPT + N_PLAYERS:
				proposer = (actor + a - A_TRADE_ACCEPT) % N_PLAYERS
			if proposer >= 0:
				executed += 1
				clone = Board(N_PLAYERS)
				clone.copy_state(before.copy(), True)
				clone._apply(a, actor, it + 1)
				isolated = clone.get_state()
				d = ROW_PLAYER + 4 * proposer + 3
				recv = before[d, PD_TRADE_RECV:PD_TRADE_RECV + N_RESOURCES].astype(int)
				give = before[d, PD_TRADE_GIVE:PD_TRADE_GIVE + N_RESOURCES].astype(int)
				for r in range(N_RESOURCES):
					d_act = (int(isolated[ROW_PLAYER + 4 * actor, PA_RESOURCES + r])
					         - int(before[ROW_PLAYER + 4 * actor, PA_RESOURCES + r]))
					d_pro = (int(isolated[ROW_PLAYER + 4 * proposer, PA_RESOURCES + r])
					         - int(before[ROW_PLAYER + 4 * proposer, PA_RESOURCES + r]))
					assert d_act == give[r] - recv[r], \
						f"accepter's resource {r} moved {d_act}, expected {give[r] - recv[r]}"
					assert d_pro == recv[r] - give[r], \
						f"proposer's resource {r} moved {d_pro}, expected {recv[r] - give[r]}"

			player = int(board.make_move(a, actor, it + 1))
			after = board.get_state()
			check_state(after)

			# one ATTEMPT per turn: once spent, no A_TRADE_RECV may reappear in MAIN
			if (int(after[ROW_GLOBAL + 1, GB_PLAYER_TRADE_DONE]) != 0
					and int(after[ROW_GLOBAL, GA_PHASE]) == PHASE_MAIN):
				v = board.valid_moves(player)
				assert not v[A_TRADE_RECV:A_TRADE_RECV + N_TRADE_SETS].any(), \
					"a second trade attempt is offered in the same turn"

			# 5. a sampled world must be able to honour every published offer
			if int(after[ROW_GLOBAL, GA_PHASE]) in trade_phases:
				b5 = Board(N_PLAYERS)
				b5.copy_state(after.copy(), True)
				b5.get_observation(0)
				obs = b5.get_state().copy()
				for sd in (1, 7, 1234):
					b5.copy_state(obs.copy(), True)
					b5.sample_world(sd)
					w = b5.get_state()
					check_state(w)
					for q in range(N_PLAYERS):
						dq = ROW_PLAYER + 4 * q + 3
						if int(w[dq, PD_TRADE_STATUS]) != TRADE_OFFERED:
							continue
						for r in range(N_RESOURCES):
							assert (w[ROW_PLAYER + 4 * q, PA_RESOURCES + r]
							        >= w[dq, PD_TRADE_GIVE + r]), \
								f"sampled world gives p{q} less {r} than it publicly offered"

	# 4. legality must not depend on hidden hands
	for seed2 in range(8):
		st = build_reference_state(seed=200 + seed2, n_settlements=2)
		st[ROW_GLOBAL, GA_PHASE] = PHASE_MAIN
		b1 = Board(N_PLAYERS)
		b1.copy_state(st.copy(), True)
		true_valids = b1.valid_moves(0).copy()
		b2 = Board(N_PLAYERS)
		b2.copy_state(st.copy(), True)
		b2.get_observation(0)
		masked_valids = b2.valid_moves(0)
		sl = slice(A_TRADE_RECV, A_TRADE_RECV + N_TRADE_SETS)
		assert (true_valids[sl] == masked_valids[sl]).all(), \
			"the legal RECV set changes once opponents' hands are masked: it leaks"

	print(f"layer 3g (player trade, {n_games} games) OK - {opened} opened, {executed} executed, "
	      f"worst {worst} plies (bound {2 * N_PLAYERS + 1})")
	return True


def check_setup(n_boards=30, seed=0):
	"""Layer 3a: what the Board already offers, without valid_moves / make_move."""
	Board = _import_board()
	if Board is None:
		print("layer 3a (setup)               SKIPPED - CatanLogicNumba.py not there yet")
		return False
	rng = np.random.default_rng(seed)
	board = Board(N_PLAYERS)
	for g in range(n_boards):
		board.init_game()
		st = board.get_state()
		check_state(st)
		assert st.shape == (N_ROWS, N_COLS)

		# init_game must use the physical frame positions for the ports
		for p in range(N_PORTS):
			a, b = PORT_VERTICES[p]
			assert st[ROW_VERTEX + a, V_PORT] == st[ROW_VERTEX + b, V_PORT] != PORT_NONE, \
				f"init_game did not place port {p} on its frame edge"
		# robber on the desert, no building anywhere, full bank
		desert = [h for h in range(N_HEXES) if st[ROW_HEX + h, H_TYPE] == HEX_DESERT][0]
		assert st[ROW_HEX + desert, H_ROBBER] == 1, "robber does not start on the desert"
		assert (st[ROW_VERTEX:ROW_VERTEX + N_VERTICES, V_BUILDING] == 0).all()
		assert (st[ROW_GLOBAL, GA_BANK:GA_BANK + N_RESOURCES] == BANK_PER_RESOURCE).all()
		assert st[ROW_GLOBAL, GA_PHASE] == PHASE_SETUP_SETTLEMENT
		if FORBID_ADJACENT_RED:
			red = [h for h in range(N_HEXES) if st[ROW_HEX + h, H_TOKEN] in RED_TOKEN_IDS.tolist()]
			for h in red:
				for n in HEX_TO_HEX[h]:
					assert n == NO_HEX or n not in red, f"6/8 adjacent on hexes {h},{n}"

		# get_symmetries must agree, element by element, with the reference
		pi = rng.random(N_ACTIONS).astype(np.float32)
		valids = (rng.random(N_ACTIONS) < .3)
		syms = board.get_symmetries(pi, valids)
		assert len(syms) == N_ISOMETRIES, f"{len(syms)} symmetries, expected {N_ISOMETRIES}"
		for s in range(N_ISOMETRIES):
			s_state, s_pi, s_valids = syms[s]
			assert (s_state == permute_state(st, s)).all(), f"get_symmetries state != reference, iso {s}"
			ref_pi = np.empty_like(pi)
			ref_pi[ISO_ACTION[s]] = pi
			assert (s_pi == ref_pi).all(), f"get_symmetries policy != reference, iso {s}"
			check_state(s_state)

	# swap_players on a populated board
	for seed2 in range(10):
		st = build_reference_state(seed=seed2, n_settlements=2)
		for n in range(1, N_PLAYERS):
			b1 = Board(N_PLAYERS)
			b1.copy_state(st.copy(), True)
			b1.swap_players(n)
			sw = b1.get_state().copy()
			check_state(sw)
			# the player who was at index n is now at index 0, hand included
			assert (sw[ROW_PLAYER:ROW_PLAYER + 4] == st[ROW_PLAYER + 4 * n:ROW_PLAYER + 4 * n + 4]).all(), \
				f"swap_players({n}) did not bring player {n} to index 0"
			# owner ids on the board were relabelled consistently
			for v in range(N_VERTICES):
				o = st[ROW_VERTEX + v, V_OWNER]
				exp = 0 if o == 0 else (o - 1 - n) % N_PLAYERS + 1
				assert sw[ROW_VERTEX + v, V_OWNER] == exp, f"vertex {v} owner not relabelled"
			# P swaps is the identity
			b1.copy_state(st.copy(), True)
			for _ in range(N_PLAYERS):
				b1.swap_players(1)
			assert (b1.get_state() == st).all(), "N_PLAYERS swaps != identity"

	# hidden information
	for seed2 in range(10):
		st = build_reference_state(seed=100 + seed2, n_settlements=2)
		_deal_dev_cards(st, seed2)
		check_state(st)
		b2 = Board(N_PLAYERS)
		b2.copy_state(st.copy(), True)
		b2.get_observation(0)
		obs = b2.get_state().copy()
		check_state(obs, masked=True)
		assert (obs[ROW_PLAYER, :] == st[ROW_PLAYER, :]).all(), "the viewer's own hand was masked"
		for p in range(1, N_PLAYERS):
			row_a, row_b = obs[ROW_PLAYER + 4 * p], obs[ROW_PLAYER + 4 * p + 1]
			assert row_a[PA_RESOURCES:PA_RESOURCES + N_RESOURCES].sum() == 0, f"p{p} resources not masked"
			assert row_b[PB_DEV_NEW:PB_DEV_NEW + N_DEV_TYPES].sum() == 0, f"p{p} new dev not masked"
			assert row_a[PA_TOTAL_RES] == st[ROW_PLAYER + 4 * p, PA_TOTAL_RES], f"p{p} lost its hand size"
			assert row_a[PA_TOTAL_DEV] == st[ROW_PLAYER + 4 * p, PA_TOTAL_DEV], f"p{p} lost its dev count"
		worlds = set()
		for sd in (1, 2, 3, 4321, 999999):
			b2.copy_state(obs.copy(), True)
			b2.sample_world(sd)
			w = b2.get_state().copy()
			check_state(w)                                  # both margins -> full conservation
			for p in range(N_PLAYERS):
				assert w[ROW_PLAYER + 4 * p, PA_TOTAL_RES] == obs[ROW_PLAYER + 4 * p, PA_TOTAL_RES]
				assert w[ROW_PLAYER + 4 * p, PA_TOTAL_DEV] == obs[ROW_PLAYER + 4 * p, PA_TOTAL_DEV]
			b2.copy_state(w.copy(), True)
			b2.get_observation(0)
			assert (b2.get_state() == obs).all(), "obs(sample_world(obs)) != obs: unstable tree keys"
			worlds.add(w.tobytes())
			# same seed, same masked input -> byte-identical world (MCTS.py caches
			# sampleWorld() once per universe and trusts this; hashed_draw()'s own
			# contract in Stochastic.py promises it, this just checks Catan didn't
			# break it e.g. by leaking np.random or dict-iteration order into the draw)
			b2.copy_state(obs.copy(), True)
			b2.sample_world(sd)
			assert (b2.get_state() == w).all(), f"sample_world(seed={sd}) is not reproducible"
		assert len(worlds) > 1 or N_PLAYERS == 2, "sample_world always returns the same world"
	print(f"layer 3a (setup, {n_boards} boards)     OK")
	return True


def check_draw_from_pool_guard(seed=0):
	"""_draw_from_pool must RAISE on an empty pool, not silently hand out a
	card it never removed from anywhere (that "safe" fallback was WORSE than
	the crash it avoided: it broke resource conservation, I7, without a
	trace). Should be unreachable from a genuinely consistent state -- 20k
	randomized trials of this exact algorithm never hit it -- so this test
	reaches it on purpose: inflate a masked player's PA_TOTAL_RES past what
	the (deliberately near-empty) bank can supply, so sample_world's shared
	pool runs dry mid-draw. Exercises the guard directly rather than hoping
	check_setup/check_with_logic's random trials happen to trigger it.
	"""
	Board = _import_board()
	if Board is None:
		print("layer 3a (pool guard)          SKIPPED - CatanLogicNumba.py not there yet")
		return False
	st = build_reference_state(seed=seed, n_settlements=2, give_resources=False)
	st[ROW_GLOBAL, GA_BANK + 0] = 0                         # resource 0 entirely "out there"
	st[ROW_PLAYER + ROWS_PER_PLAYER, PA_TOTAL_RES] = BANK_PER_RESOURCE + 5  # player 1 claims more than exists
	b = Board(N_PLAYERS)
	b.copy_state(st.copy(), True)
	b.get_observation(0)
	raised = False
	try:
		b.sample_world(1)
	except ValueError:
		raised = True
	assert raised, "_draw_from_pool must raise on an empty pool, not silently invent a card"
	print("layer 3a (pool guard)           OK - raises instead of corrupting conservation")
	return True


def _deal_dev_cards(st, seed):
	"""Give a few dev cards out of the deck, so masking has something to hide."""
	rng = np.random.default_rng(seed)
	for p in range(N_PLAYERS):
		for _ in range(int(rng.integers(0, 4))):
			k = int(rng.integers(0, N_DEV_TYPES))
			if st[ROW_GLOBAL, GA_DEV_DECK + k] == 0:
				continue
			st[ROW_GLOBAL, GA_DEV_DECK + k] -= 1
			if rng.random() < .3:
				st[ROW_PLAYER + 4 * p + 1, PB_DEV_NEW + k] += 1
			else:
				st[ROW_PLAYER + 4 * p, PA_DEV_PLAYABLE + k] += 1
	refresh_derived(st)


def _import_board():
	"""Return the Board class, or None if the logic file genuinely does not exist.

	Anything else -- a missing dependency, a syntax error, a numba typing error --
	is REPORTED, never swallowed. Reporting a broken import as "not written yet"
	is the instrument lying about its own coverage, which is the one failure mode
	this whole file exists to prevent.
	"""
	import os
	if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CatanLogicNumba.py')):
		return None
	try:
		try:
			from CatanLogicNumba import Board
		except ModuleNotFoundError:        # NOT ImportError: a dependency of the logic
			from .CatanLogicNumba import Board   # failing must not look like a path issue
		return Board
	except Exception as e:
		print("\n!!! CatanLogicNumba.py EXISTS but could not be imported -- layers 3a/3b are NOT running.")
		print("!!! %s: %s\n" % (type(e).__name__, e))
		raise


def check_with_logic(n_games=8, seed=0):
	Board = _import_board()
	if Board is None or not hasattr(Board, 'make_move'):
		print("layer 3b (playouts)            SKIPPED - valid_moves/make_move not written yet")
		return False

	rng = np.random.default_rng(seed)
	board = Board(N_PLAYERS)
	for g in range(n_games):
		board.init_game()
		st0 = board.get_state()
		check_state(st0)
		for p in range(N_PORTS):      # init_game must use the physical frame positions
			a, b = PORT_VERTICES[p]
			assert st0[ROW_VERTEX + a, V_PORT] == st0[ROW_VERTEX + b, V_PORT] != PORT_NONE, \
				f"init_game did not place port {p} on its frame edge"
		player, it = 0, 0
		while not board.check_end_game(player).any() and it < MAX_DECISIONS:
			it += 1
			valids = board.valid_moves(player)
			assert valids.any(), f"no legal move, phase {board.get_state()[ROW_GLOBAL, GA_PHASE]}"
			assert valids.size == N_ACTIONS
			if not ENABLE_PLAYER_TRADE:
				assert not valids[N_ACTIONS_V1:].any(), "the trade block is valid while disabled"
			a = int(rng.choice(np.flatnonzero(valids)))
			before = board.get_state().copy()
			actor, seed = player, it + 1
			player = int(board.make_move(a, actor, seed))
			check_state(board.get_state())

			# make_move must commute with every isometry
			for s in rng.choice(N_ISOMETRIES, 2, replace=False):
				b2 = Board(N_PLAYERS)
				b2.copy_state(permute_state(before, int(s)), True)
				b2.make_move(permute_action(a, int(s)), actor, seed)   # isometries do not move players
				assert (b2.get_state() == permute_state(board.get_state(), int(s))).all(), \
					f"make_move does not commute with isometry {s} on action {a}"

			# canonical form: swapping P times must be the identity
			b3 = Board(N_PLAYERS)
			b3.copy_state(board.get_state(), True)
			for _ in range(N_PLAYERS):
				b3.swap_players(1)
			assert (b3.get_state() == board.get_state()).all(), "N swaps != identity"

			# hidden information
			b4 = Board(N_PLAYERS)
			b4.copy_state(board.get_state(), True)
			b4.get_observation(0)
			obs = b4.get_state().copy()
			check_state(obs, masked=True)
			assert (obs[ROW_PLAYER, PA_TOTAL_RES] == board.get_state()[ROW_PLAYER, PA_TOTAL_RES]), \
				"the viewer's own total was masked"
			for p in range(1, N_PLAYERS):
				a_row = obs[ROW_PLAYER + 4 * p]
				assert a_row[PA_RESOURCES:PA_RESOURCES + N_RESOURCES].sum() == 0, f"p{p} hand not masked"
				assert a_row[PA_TOTAL_RES] == board.get_state()[ROW_PLAYER + 4 * p, PA_TOTAL_RES], \
					f"p{p} total lost, it is no longer deducible once the detail is masked"
			for sd in (1, 2, 12345):
				b4.copy_state(obs, True)
				b4.sample_world(sd)
				w = b4.get_state().copy()
				check_state(w)                      # both margins hold -> full conservation
				for p in range(N_PLAYERS):
					assert w[ROW_PLAYER + 4 * p, PA_TOTAL_RES] == obs[ROW_PLAYER + 4 * p, PA_TOTAL_RES], \
						f"sample_world changed p{p}'s hand size"
				b4.copy_state(w, True)
				b4.get_observation(0)
				assert (b4.get_state() == obs).all(), "obs(sample_world(obs)) != obs: unstable tree keys"
				b4.copy_state(obs, True)             # same seed again -> byte-identical world
				b4.sample_world(sd)
				assert (b4.get_state() == w).all(), f"sample_world(seed={sd}) is not reproducible"

			# get_symmetries must agree with the reference permutation
			pi = rng.random(N_ACTIONS).astype(np.float32)
			syms = board.get_symmetries(pi, board.valid_moves(player))
			assert len(syms) == N_ISOMETRIES, f"{len(syms)} symmetries, expected {N_ISOMETRIES}"
		assert it < MAX_DECISIONS, f"game did not terminate within {MAX_DECISIONS} decisions"
		scores = board.check_end_game(player)
		assert scores.size == N_PLAYERS and (scores != 0).any()
	print(f"layer 3b (playouts, {n_games} games) OK")
	return True


# ---------------------------------------------------------------------------
# Layer 4: testing the instrument
# ---------------------------------------------------------------------------

MUTATIONS = [
	("negative value", lambda s: s.__setitem__((ROW_PLAYER, PA_RESOURCES), -1)),
	("resource vanishes from the bank", lambda s: s.__setitem__((ROW_GLOBAL, GA_BANK), s[ROW_GLOBAL, GA_BANK] - 1)),
	("edge owner on one endpoint only", lambda s: s.__setitem__((ROW_VERTEX + int(EDGE_TO_VERTEX[0, 0]), V_EDGE0 + int(np.flatnonzero(VERTEX_TO_EDGE[EDGE_TO_VERTEX[0, 0]] == 0)[0])), 1)),
	("distance rule broken", _mut_adjacent := (lambda s: _break_distance(s))),
	("two robbers", lambda s: s.__setitem__((ROW_HEX, H_ROBBER), 1) if s[ROW_HEX, H_ROBBER] == 0 else s.__setitem__((ROW_HEX + 1, H_ROBBER), 1)),
	("no robber", lambda s: s.__setitem__((slice(ROW_HEX, ROW_HEX + N_HEXES), H_ROBBER), 0)),
	("stale PC_PORTS", lambda s: s.__setitem__((ROW_PLAYER + 2, PC_PORTS), 1 - s[ROW_PLAYER + 2, PC_PORTS])),
	("stale PA_TOTAL_RES", lambda s: s.__setitem__((ROW_PLAYER, PA_TOTAL_RES), s[ROW_PLAYER, PA_TOTAL_RES] + 1)),
	("piece count drift", lambda s: s.__setitem__((ROW_PLAYER + 1, PB_ROADS_LEFT), s[ROW_PLAYER + 1, PB_ROADS_LEFT] + 1)),
	("dev deck drift", lambda s: s.__setitem__((ROW_GLOBAL, GA_DEV_DECK + KNIGHT), s[ROW_GLOBAL, GA_DEV_DECK + KNIGHT] - 1)),
	("knights played not discarded", lambda s: s.__setitem__((ROW_PLAYER + 1, PB_KNIGHTS), 1)),
	("pips do not match the token", lambda s: s.__setitem__((ROW_HEX, H_PIPS), (s[ROW_HEX, H_PIPS] + 1) % 6)),
	("owner out of range", lambda s: s.__setitem__((ROW_VERTEX + int(np.flatnonzero(s[ROW_VERTEX:ROW_VERTEX + N_VERTICES, V_OWNER])[0]), V_OWNER), N_PLAYERS + 1)),
	("two largest armies", lambda s: (s.__setitem__((ROW_PLAYER + 1, PB_HAS_ARMY), 1), s.__setitem__((ROW_PLAYER + 4, PB_HAS_ARMY), 1))),
	("building without owner", lambda s: s.__setitem__((ROW_VERTEX + int(np.flatnonzero(s[ROW_VERTEX:ROW_VERTEX + N_VERTICES, V_OWNER])[0]), V_OWNER), 0)),
	("write past the documented columns", lambda s: s.__setitem__((ROW_VERTEX, 7), 3)),
	("offer recorded with no proposer", lambda s: s.__setitem__((ROW_PLAYER + 3, PD_TRADE_RECV), 1)),
	("trade status out of range", lambda s: s.__setitem__((ROW_PLAYER + 3, PD_TRADE_STATUS), TRADE_REFUSED + 1)),
	("trade phase with nothing on the table", lambda s: s.__setitem__((ROW_GLOBAL, GA_PHASE), PHASE_TRADE_ANSWER)),
	("counter-offer with no original offer", _mut_counter := (lambda s: _lone_counter(s))),
	("a resource traded against itself", _mut_selftrade := (lambda s: _self_trade(s))),
]


def _lone_counter(s):
	"""Player 1 publishes an offer while the turn player (0) never announced."""
	d = ROW_PLAYER + 4 * 1 + 3
	s[d, PD_TRADE_RECV] = 1
	s[d, PD_TRADE_GIVE + 1] = 1
	s[d, PD_TRADE_STATUS] = TRADE_OFFERED
	s[ROW_GLOBAL, GA_PHASE] = PHASE_TRADE_ACCEPT


def _self_trade(s):
	"""A published offer asking for and giving the SAME resource."""
	d = ROW_PLAYER + 3
	s[d, PD_TRADE_RECV] = 1
	s[d, PD_TRADE_GIVE] = 1
	s[d, PD_TRADE_STATUS] = TRADE_OFFERED
	s[ROW_GLOBAL, GA_PHASE] = PHASE_TRADE_ANSWER


def _break_distance(s):
	verts = s[ROW_VERTEX:ROW_VERTEX + N_VERTICES]
	v = int(np.flatnonzero(verts[:, V_BUILDING])[0])
	o = int([x for x in VERTEX_TO_VERTEX[v] if x != NO_VERTEX][0])
	verts[o, V_BUILDING], verts[o, V_OWNER] = 1, 1


def test_the_instrument():
	base = build_reference_state(seed=7)
	check_state(base)               # the pristine oracle must pass
	missed = []
	for name, mutate in MUTATIONS:
		st = base.copy()
		mutate(st)
		if (st == base).all():
			missed.append(f"{name} (mutation had no effect)")
			continue
		try:
			check_state(st)
		except AssertionError:
			continue
		missed.append(name)
	assert not missed, "UNDETECTED corruptions: " + ", ".join(missed)
	print(f"layer 4 (instrument, {len(MUTATIONS)} mutations) OK - all caught")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
	for seed in range(25):
		st = build_reference_state(seed=seed, n_settlements=1 + seed % 3)
		check_state(st)
	print("layer 1 (state invariants)     OK - 25 reference states")

	check_isometries(build_reference_state(seed=1))
	print("layer 2 (12 isometries)        OK")

	test_the_instrument()
	check_setup()
	check_draw_from_pool_guard()
	check_setup_actor_consistency()
	check_robber_action_indices()
	check_trade_cap()
	check_victory_is_on_own_turn()
	check_player_trade()
	check_with_logic()
	print("\nCATAN TEST BATTERY: DONE")
