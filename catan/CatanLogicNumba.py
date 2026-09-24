import numpy as np
from numba import njit
import numba

import os
import sys

try:                                    # inside the package (main.py, pit.py)
	from .CatanConstants import *
except ImportError:                     # run directly from the catan/ folder
	from CatanConstants import *

# Stochastic.py lives at the repository root: make it importable when this
# module is used from inside catan/ (CatanTest.py run directly)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
	sys.path.append(_ROOT)
from Stochastic import hashed_draw

############################## BOARD DESCRIPTION ##############################
#
# state: np.ndarray[int8] of shape (75 + 4*N_PLAYERS, N_COLS=12), 87 x 12 for 3
# players. Column names are the constants of CatanConstants.py. No value is ever
# negative (absence is 0 or a NO_* sentinel) and no value is a bitfield;
# CatanTest.py checks both after every move.
#
# The board is a GRAPH: only vertices and hexes get a row (a network token). An
# edge has no row, its owner is stored on BOTH endpoints (V_EDGE0..2 follow
# VERTEX_TO_EDGE) and the network scores it with a bilinear head over them.
# Static incidences live in CatanConstants.py, never in the state.
#
#   rows [0 .. 53]      one per VERTEX: building, owner (canonical order), port,
#                       owner of each incident edge. Ports are stored per vertex so
#                       that the 12 isometries are plain row permutations.
#   rows [54 .. 72]     one per HEX: type, pips (economic value), token id
#                       (identity: two 8s produce together), robber.
#   4 rows per player   CANONICAL order, player 0 is the one to move:
#     A  hand (resources, playable dev cards) and the two public TOTALS
#     B  dev cards bought this turn, knights, pieces left, longest road, bonuses
#     C  VP (public / from dev cards), dev played this turn, ports (cached from
#        the vertices), dev cards bought this turn, cards owed after a 7, bank trades
#     D  standing trade offer (ask, give, status), PUBLIC. Kept on its author's
#        rows so it rotates with its owner; the protocol state (composer, next
#        responder) is derived from the statuses, never stored.
#   2 global rows
#     A  bank, dev deck per type, last roll, phase
#     B  round (lo/hi), pending roads or discards, turn player (relative to the
#        player to move), setup step, dev cards played per type, chance counter,
#        player trade done this turn
#
# HIDDEN INFORMATION (get_observation / sample_world)
#   Hidden: the resources and dev cards of the OTHER players, and the composition
#   of the dev deck. get_observation(viewer) zeroes them and keeps the public
#   totals PA_TOTAL_RES / PA_TOTAL_DEV; the deck size stays deducible
#   (dev_deck_size). A hand is masked iff its detail no longer sums to its total,
#   the deck iff it no longer sums to dev_deck_size: no flag to keep in sync.
#   sample_world() re-deals every masked hand and the deck so that all margins
#   hold: each total, each resource outside the bank, each dev type not played.
#   A published offer proves its author holds what it gives: those cards are
#   reserved before any random draw.
#   Legality never reads hidden information: valid_moves() is the same on the
#   true state and on the observation.
#   With 2 players the opponent's resources are deducible from the bank.
#
############################## ACTION DESCRIPTION #############################
#
# One flat space of N_ACTIONS ids (402 for 3 players), split into DISJOINT ranges:
# an output neuron always means exactly one thing, whatever the phase. Offsets
# are the A_* constants, never hard-coded.
#
#   A_ROAD           + e              e   in [0, 72)   build a road on edge e
#   A_SETTLEMENT     + v              v   in [0, 54)   build a settlement on vertex v
#   A_CITY           + v              v   in [0, 54)   upgrade own settlement on v
#   A_BUY_DEV                                          buy one development card
#   A_PLAY_DEV       + k              k   in [0, 2)    knight / road building. Monopoly
#                                                      and Year of Plenty have no separate
#                                                      "play" id: their parameter block IS
#                                                      the action, which saves a ply
#   A_ROBBER         + h*N_PLAYERS+t  h   in [0, 19)   move the robber to h and rob t,
#                                    t   in [0, P)     t = 0 means rob nobody,
#                                                      t >= 1 is a RELATIVE player id
#   A_ROLL                                             roll the dice. A decision only
#                                                      because a knight may be played
#                                                      first; auto-resolved otherwise
#   A_MONOPOLY       + r              r   in [0, 5)    play Monopoly on resource r
#   A_YEAR_OF_PLENTY + p              p   in [0, 15)   play Year of Plenty, YOP_PAIRS[p]
#   A_BANK_TRADE     + g*4 + k        g   in [0, 5)    give resource g, receive the
#                                    k   in [0, 4)     k-th other resource; the ratio
#                                                      (4:1, 3:1 or 2:1) is implied by
#                                                      the best port owned, never chosen
#   A_DISCARD        + r              r   in [0, 5)    discard ONE card, repeated
#   A_END_TURN
#
#   Player trade, only while ENABLE_PLAYER_TRADE:
#   A_TRADE_RECV     + s              s   in [0, 55)   announce: I ask for TRADE_SETS[s]
#   A_TRADE_GIVE     + s              s   in [0, 55)   ... and offer TRADE_SETS[s] for it
#   A_TRADE_OK                                         accept the turn player's offer
#   A_TRADE_NO                                         decline it, without countering
#   A_TRADE_ACCEPT   + t              t   in [0, P)    the turn player takes the counter
#                                                      from relative player t;
#                                                      t = 0 = refuse them all
#
#   An announcement is FACTORISED over two plies: a flat id per (ask, offer) pair
#   would need 55*55 = 3025 ids, the split needs 55+55 and loses nothing.
#
#   PROTOCOL (non-official: real haggling has no bounded ply count)
#     turn player : RECV, GIVE                                    2 plies
#     each other player, in seat order, answers ONCE, and always
#     to the TURN PLAYER's offer:
#         OK      -> executes at once, the round table is CLOSED  1 ply
#         NO      -> next responder                               1 ply
#         counter -> RECV, GIVE, stacked for the turn player,
#                    then the next responder                      2 plies
#     if at least one counter is standing:
#         turn player : A_TRADE_ACCEPT + t, or t=0 to refuse all  1 ply
#     Worst case 2P+1 plies (7 at P=3). A counter is never submitted to the other
#     responders, which is what keeps that bound; every trade involves the turn
#     player on one side.
#
# Relative player ids (A_ROBBER, A_TRADE_ACCEPT) are expressed in the CANONICAL
# frame, so that swap_players() never changes the meaning of an action id. The
# trade SET ids name resources, which no isometry permutes, so they map to
# themselves (asserted in CatanConstantsTest.py).
#
# A choice with a single legal option is NEVER exposed: make_move() resolves it
# on the spot (only one legal edge during setup, a single robbable opponent, a
# forced discard, ...). Each one exposed would cost a full MCTS search.
#
############################## SYMMETRIES #####################################
#
# The board has 12 isometries (6 rotations x mirror). get_symmetries() applies
# ISO_VERTEX / ISO_EDGE / ISO_HEX to the state AND the same permutation to the
# policy. Player and global rows are untouched.
#
###############################################################################


@njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (N_ROWS, N_COLS)


@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return N_ACTIONS


@njit(cache=True, nogil=True)
def edge_owner(vertices, e):
	return vertices[EDGE_TO_VERTEX[e, 0], V_EDGE0 + EDGE_SLOT[e, 0]]


@njit(cache=True, nogil=True)
def longest_road(vertices, player_id):
	"""Longest continuous road: the longest TRAIL (no repeated edge) in the
	player's road graph, cut at any vertex carrying an opponent's building.
	Iterative DFS -- recursion inside a jitclass method is not worth the risk."""
	best = 0
	used = np.zeros(N_EDGES, dtype=np.int8)
	stack_e = np.zeros(N_EDGES + 1, dtype=np.int16)
	stack_v = np.zeros(N_EDGES + 1, dtype=np.int16)
	stack_k = np.zeros(N_EDGES + 1, dtype=np.int8)
	for e0 in range(N_EDGES):
		if edge_owner(vertices, e0) != player_id:
			continue
		for start in range(2):
			used[:] = 0
			used[e0] = 1
			depth = 0
			stack_e[0], stack_v[0], stack_k[0] = e0, EDGE_TO_VERTEX[e0, start], 0
			if best < 1:
				best = 1
			while depth >= 0:
				e, v, k = stack_e[depth], stack_v[depth], stack_k[depth]
				if k >= 3:
					used[e] = 0
					depth -= 1
					continue
				stack_k[depth] = k + 1
				if vertices[v, V_BUILDING] != 0 and vertices[v, V_OWNER] != player_id:
					continue                        # an opponent's building cuts the road
				e2 = VERTEX_TO_EDGE[v, k]
				if e2 == NO_EDGE or used[e2] != 0 or edge_owner(vertices, e2) != player_id:
					continue
				v2 = EDGE_TO_VERTEX[e2, 0] if EDGE_TO_VERTEX[e2, 1] == v else EDGE_TO_VERTEX[e2, 1]
				used[e2] = 1
				depth += 1
				stack_e[depth], stack_v[depth], stack_k[depth] = e2, v2, 0
				if depth + 1 > best:
					best = depth + 1
	return best


spec = [
	('num_players'  , numba.int8),
	('max_rounds'   , numba.int16),
	('score_win'    , numba.int8),

	('state'        , numba.int8[:,:]),
	('vertices'     , numba.int8[:,:]),   # 54 rows, see BOARD DESCRIPTION
	('hexes'        , numba.int8[:,:]),   # 19 rows
	('players'      , numba.int8[:,:]),   # 4 rows per player, see BOARD DESCRIPTION
	('globals_'     , numba.int8[:,:]),   # 2 rows
]
@numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.num_players = num_players
		self.max_rounds = MAX_ROUNDS
		self.score_win = VP_TO_WIN
		self.state = np.zeros((75 + 4*num_players, N_COLS), dtype=np.int8)
		self.init_game()

	def get_state(self):
		return self.state

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state
		n = self.num_players
		self.vertices = self.state[0                       :N_VERTICES              , :]   # 54
		self.hexes    = self.state[N_VERTICES              :N_VERTICES+N_HEXES      , :]   # 19
		self.players  = self.state[N_VERTICES+N_HEXES      :N_VERTICES+N_HEXES+4*n  , :]   # 4*N
		self.globals_ = self.state[N_VERTICES+N_HEXES+4*n  :N_VERTICES+N_HEXES+4*n+2, :]   # 2

	############################## EDGE ACCESS ################################
	# An edge has no row: its owner lives on BOTH endpoints and the two copies
	# must never diverge. Nothing outside these two helpers touches V_EDGE*.

	def _edge_owner(self, e):
		v = EDGE_TO_VERTEX[e, 0]
		for k in range(3):
			if VERTEX_TO_EDGE[v, k] == e:
				return self.vertices[v, V_EDGE0 + k]
		return np.int8(0)

	def _set_edge(self, e, owner):
		for i in range(2):
			v = EDGE_TO_VERTEX[e, i]
			for k in range(3):
				if VERTEX_TO_EDGE[v, k] == e:
					self.vertices[v, V_EDGE0 + k] = owner

	############################## SETUP ######################################

	def init_game(self):
		# a FRESH array: copy_state returns early when handed the very array it
		# already wraps, which would leave the views unbuilt
		self.copy_state(np.zeros((75 + 4*self.num_players, N_COLS), dtype=np.int8), False)

		# --- hexes: types, then number tokens -----------------------------
		types = HEX_DISTRIBUTION.copy()
		tokens = TOKEN_DISTRIBUTION.copy()
		if RANDOM_BOARD:
			for i in range(types.size - 1, 0, -1):
				j = np.random.randint(0, i + 1)
				types[i], types[j] = types[j], types[i]
			# official rule: no two red tokens (6 or 8) on adjacent hexes.
			# Rejection sampling; the bound keeps init_game total even if the
			# rule were ever made unsatisfiable.
			for _attempt in range(1000):
				for i in range(tokens.size - 1, 0, -1):
					j = np.random.randint(0, i + 1)
					tokens[i], tokens[j] = tokens[j], tokens[i]
				if not FORBID_ADJACENT_RED or self._token_layout_ok(types, tokens):
					break

		t = 0
		for h in range(N_HEXES):
			self.hexes[h, H_TYPE] = types[h]
			if types[h] == HEX_DESERT:
				self.hexes[h, H_TOKEN] = 0
				self.hexes[h, H_ROBBER] = 1          # the robber starts on the desert
			else:
				self.hexes[h, H_TOKEN] = tokens[t]
				t += 1
			self.hexes[h, H_PIPS] = TOKEN_PIPS[self.hexes[h, H_TOKEN]]

		# --- ports: fixed frame positions, shuffled types ------------------
		ports = PORT_DISTRIBUTION.copy()
		if RANDOM_BOARD:
			for i in range(ports.size - 1, 0, -1):
				j = np.random.randint(0, i + 1)
				ports[i], ports[j] = ports[j], ports[i]
		for p in range(N_PORTS):
			for i in range(2):
				self.vertices[PORT_VERTICES[p, i], V_PORT] = ports[p]

		# --- players and globals -------------------------------------------
		for p in range(self.num_players):
			self.players[4*p + 1, PB_SETTLEMENTS_LEFT] = MAX_SETTLEMENTS
			self.players[4*p + 1, PB_CITIES_LEFT] = MAX_CITIES
			self.players[4*p + 1, PB_ROADS_LEFT] = MAX_ROADS
		for r in range(N_RESOURCES):
			self.globals_[0, GA_BANK + r] = BANK_PER_RESOURCE
		for k in range(N_DEV_TYPES):
			self.globals_[0, GA_DEV_DECK + k] = DEV_DISTRIBUTION[k]
		self.globals_[0, GA_PHASE] = PHASE_SETUP_SETTLEMENT

	def _token_layout_ok(self, types, tokens):
		# map token index -> hex index, skipping the desert
		red = np.zeros(N_HEXES, dtype=np.int8)
		t = 0
		for h in range(N_HEXES):
			if types[h] == HEX_DESERT:
				continue
			if tokens[t] == RED_TOKEN_IDS[0] or tokens[t] == RED_TOKEN_IDS[1]:
				red[h] = 1
			t += 1
		for h in range(N_HEXES):
			if red[h] == 0:
				continue
			for i in range(6):
				n = HEX_TO_HEX[h, i]
				if n != NO_HEX and red[n] == 1:
					return False
		return True

	############################## SCORING ####################################

	def get_score(self, player):
		return self.players[4*player + 2, PC_VP_PUBLIC] + self.players[4*player + 2, PC_VP_DEV]

	def get_round(self):
		# int16: GB_ROUND_HI * 100 overflows int8 as soon as the round reaches 200
		return np.int16(self.globals_[1, GB_ROUND_HI]) * np.int16(100) + np.int16(self.globals_[1, GB_ROUND_LO])

	def _set_round(self, r):
		self.globals_[1, GB_ROUND_HI] = r // 100
		self.globals_[1, GB_ROUND_LO] = r % 100

	def check_end_game(self, next_player):
		# OFFICIAL RULE: a player wins only during its OWN turn, the instant the
		# condition is met. Testing only the turn player also keeps terminality
		# from depending on the hidden VP cards of the other players, except in
		# the phases where the actor is not the turn player (discard, trade answer).
		turn_player = int(self.globals_[1, GB_TURN_PLAYER])
		timeout = self.get_round() >= self.max_rounds
		if not timeout:
			if self.get_score(turn_player) < self.score_win:
				return np.zeros(self.num_players, dtype=np.float32)
			out = np.full(self.num_players, -1., dtype=np.float32)
			out[turn_player] = 1.
			return out
		# MAX_ROUNDS timeout: nobody won on their turn, so rank on score instead
		scores = np.zeros(self.num_players, dtype=np.float32)
		for p in range(self.num_players):
			scores[p] = self.get_score(p)
		score_max = scores.max()
		who_has_won = (scores == score_max)
		several = (who_has_won.sum() > 1)
		return np.where(who_has_won, 0.01 if several else 1., -1.).astype(np.float32)

	############################## CANONICAL FORM #############################
	# swap_players(n): old player n becomes index 0, i.e. new[i] = old[(i+n)%P].
	# Owner ids stored on the board are absolute in the CURRENT frame, so they
	# have to be relabelled too -- forgetting that is the classic silent bug.

	def swap_players(self, nb_swaps):
		n = nb_swaps % self.num_players
		if n == 0:
			return
		p = self.num_players

		players_copy = self.players.copy()
		for i in range(p):
			for row in range(4):
				self.players[4*i + row, :] = players_copy[4*((i + n) % p) + row, :]

		# relabel: owner id o (1..P) refers to old index o-1 -> new index o-1-n
		relabel = np.zeros(p + 1, dtype=np.int8)
		for o in range(1, p + 1):
			relabel[o] = (o - 1 - n) % p + 1
		for v in range(N_VERTICES):
			self.vertices[v, V_OWNER] = relabel[self.vertices[v, V_OWNER]]
			for k in range(3):
				self.vertices[v, V_EDGE0 + k] = relabel[self.vertices[v, V_EDGE0 + k]]

		self.globals_[1, GB_TURN_PLAYER] = (self.globals_[1, GB_TURN_PLAYER] - n) % p

	############################## SYMMETRIES #################################

	def get_symmetries(self, policy, valid_actions):
		symmetries = [(self.state.copy(), policy.copy(), valid_actions.copy())]
		for s in range(1, N_ISOMETRIES):
			new_state = self.state.copy()
			for v in range(N_VERTICES):
				nv = ISO_VERTEX[s, v]
				new_state[nv, V_BUILDING] = self.vertices[v, V_BUILDING]
				new_state[nv, V_OWNER] = self.vertices[v, V_OWNER]
				new_state[nv, V_PORT] = self.vertices[v, V_PORT]
				for k in range(3):
					if VERTEX_TO_EDGE[v, k] != NO_EDGE:
						new_state[nv, V_EDGE0 + ISO_SLOT[s, v, k]] = self.vertices[v, V_EDGE0 + k]
			for h in range(N_HEXES):
				new_state[N_VERTICES + ISO_HEX[s, h], :] = self.hexes[h, :]
			new_policy = policy.copy()
			new_valids = valid_actions.copy()
			for a in range(N_ACTIONS):
				new_policy[ISO_ACTION[s, a]] = policy[a]
				new_valids[ISO_ACTION[s, a]] = valid_actions[a]
			symmetries.append((new_state, new_policy, new_valids))
		return symmetries

	############################## HIDDEN INFORMATION #########################
	# See HIDDEN INFORMATION in the header.

	def dev_deck_size(self):
		# public: 25 minus the dev cards in hand (public totals) and those played
		n = 0
		for k in range(N_DEV_TYPES):
			n += DEV_DISTRIBUTION[k] - self.globals_[1, GB_DEV_PLAYED + k]
		for p in range(self.num_players):
			n -= self.players[4*p, PA_TOTAL_DEV]
		return n

	def get_observation(self, viewer):
		for p in range(self.num_players):
			if p == viewer:
				continue
			for r in range(N_RESOURCES):
				self.players[4*p, PA_RESOURCES + r] = 0
			for k in range(N_DEV_TYPES):
				self.players[4*p, PA_DEV_PLAYABLE + k] = 0
				self.players[4*p + 1, PB_DEV_NEW + k] = 0
			self.players[4*p + 2, PC_VP_DEV] = 0      # derived from the dev hand
		for k in range(N_DEV_TYPES):
			self.globals_[0, GA_DEV_DECK + k] = 0

	def sample_world(self, random_seed):
		counter = np.int64(0)

		# --- resources: deal from the pool of everything outside the bank ---
		# `masked` is settled before anything is dealt: afterwards "detail !=
		# total" would also mean "not finished yet".
		pool = np.zeros(N_RESOURCES, dtype=np.int8)
		for r in range(N_RESOURCES):
			pool[r] = BANK_PER_RESOURCE - self.globals_[0, GA_BANK + r]
		masked = np.zeros(self.num_players, dtype=np.int8)
		for p in range(self.num_players):
			detail = 0
			for r in range(N_RESOURCES):
				detail += self.players[4*p, PA_RESOURCES + r]
			if detail == self.players[4*p, PA_TOTAL_RES]:      # visible hand
				for r in range(N_RESOURCES):
					pool[r] -= self.players[4*p, PA_RESOURCES + r]
			else:
				masked[p] = 1

		# Cards promised by published offers are reserved in a pass of their own,
		# before any random draw: otherwise one player's random cards could use
		# up a resource another player has publicly committed to.
		forced = np.zeros(self.num_players, dtype=np.int8)
		for p in range(self.num_players):
			if masked[p] == 0 or self.players[4*p + 3, PD_TRADE_STATUS] != TRADE_OFFERED:
				continue
			for r in range(N_RESOURCES):
				g = self.players[4*p + 3, PD_TRADE_GIVE + r]
				if g > 0:
					if pool[r] < g:
						raise ValueError("sample_world: published offers exceed the pool")
					pool[r] -= g
					self.players[4*p, PA_RESOURCES + r] += g
					forced[p] += g

		for p in range(self.num_players):
			if masked[p] == 0:
				continue
			for _c in range(self.players[4*p, PA_TOTAL_RES] - forced[p]):
				counter += 1
				r = self._draw_from_pool(pool, random_seed, counter)
				self.players[4*p, PA_RESOURCES + r] += 1

		# --- dev cards: masked hands, then the deck gets what is left ---------
		deck = 0
		for k in range(N_DEV_TYPES):
			deck += self.globals_[0, GA_DEV_DECK + k]
		deck_masked = deck != self.dev_deck_size()
		dpool = np.zeros(N_DEV_TYPES, dtype=np.int8)
		for k in range(N_DEV_TYPES):
			dpool[k] = DEV_DISTRIBUTION[k] - self.globals_[1, GB_DEV_PLAYED + k]
			if not deck_masked:
				dpool[k] -= self.globals_[0, GA_DEV_DECK + k]
		for p in range(self.num_players):
			detail = 0
			for k in range(N_DEV_TYPES):
				detail += self.players[4*p, PA_DEV_PLAYABLE + k] + self.players[4*p + 1, PB_DEV_NEW + k]
			if detail == self.players[4*p, PA_TOTAL_DEV]:
				for k in range(N_DEV_TYPES):
					dpool[k] -= self.players[4*p, PA_DEV_PLAYABLE + k] + self.players[4*p + 1, PB_DEV_NEW + k]
		for p in range(self.num_players):
			detail = 0
			for k in range(N_DEV_TYPES):
				detail += self.players[4*p, PA_DEV_PLAYABLE + k] + self.players[4*p + 1, PB_DEV_NEW + k]
			if detail == self.players[4*p, PA_TOTAL_DEV]:
				continue
			nb_new = self.players[4*p + 2, PC_TOTAL_DEV_NEW]
			for c in range(self.players[4*p, PA_TOTAL_DEV]):
				counter += 1
				k = self._draw_from_pool(dpool, random_seed, counter)
				if c < nb_new:
					self.players[4*p + 1, PB_DEV_NEW + k] += 1
				else:
					self.players[4*p, PA_DEV_PLAYABLE + k] += 1
			self.players[4*p + 2, PC_VP_DEV] = (self.players[4*p, PA_DEV_PLAYABLE + VICTORY_POINT]
			                                    + self.players[4*p + 1, PB_DEV_NEW + VICTORY_POINT])
		if deck_masked:
			for k in range(N_DEV_TYPES):
				self.globals_[0, GA_DEV_DECK + k] = dpool[k]

	def _draw_from_pool(self, pool, random_seed, counter):
		# uniform draw without replacement from a multiset. The margins hold by
		# construction, so an empty pool can only come from an inconsistent state:
		# raise rather than hand out a card that comes from nowhere
		# (exercised by CatanTest.check_draw_from_pool_guard)
		total = 0
		for i in range(pool.size):
			total += pool[i]
		if total <= 0:
			raise ValueError("_draw_from_pool: empty pool, inconsistent state")
		if random_seed == 0:
			r = np.random.randint(0, total)
		else:
			r = hashed_draw(random_seed, counter, total)
		acc = 0
		for i in range(pool.size):
			acc += pool[i]
			if r < acc:
				pool[i] -= 1
				return i
		return pool.size - 1     # unreachable when the margins are consistent

	############################## RULES: QUERIES #############################

	def _is_blocked(self, v, player):
		# an opponent's building cuts a road at vertex v
		return self.vertices[v, V_BUILDING] != 0 and self.vertices[v, V_OWNER] != player + 1

	def _can_settle(self, v):
		if self.vertices[v, V_BUILDING] != 0:
			return False
		for k in range(3):
			o = VERTEX_TO_VERTEX[v, k]
			if o != NO_VERTEX and self.vertices[o, V_BUILDING] != 0:
				return False
		return True

	def _road_connected(self, e, player):
		for i in range(2):
			v = EDGE_TO_VERTEX[e, i]
			if self.vertices[v, V_OWNER] == player + 1:
				return True
			if self._is_blocked(v, player):
				continue
			for k in range(3):
				e2 = VERTEX_TO_EDGE[v, k]
				if e2 != NO_EDGE and e2 != e and self._edge_owner(e2) == player + 1:
					return True
		return False

	def _settlement_connected(self, v, player):
		for k in range(3):
			e = VERTEX_TO_EDGE[v, k]
			if e != NO_EDGE and self._edge_owner(e) == player + 1:
				return True
		return False

	def _can_afford(self, player, cost):
		for r in range(N_RESOURCES):
			if self.players[4*player, PA_RESOURCES + r] < cost[r]:
				return False
		return True

	def _trade_ratio(self, player, give):
		if self.players[4*player + 2, PC_PORTS + give] != 0:
			return PORT_RATIO_SPECIFIC
		if self.players[4*player + 2, PC_PORTS + 5] != 0:
			return PORT_RATIO_GENERIC
		return BANK_RATIO

	def _has_any_road(self, player):
		for e in range(N_EDGES):
			if self._edge_owner(e) == 0 and self._road_connected(e, player):
				return True
		return False

	############################## RULES: PLAYER TRADE ########################
	# Legality NEVER reads a hidden hand, which is what keeps the action space
	# usable under masking (get_observation):
	#   - what I may ASK for is bounded by what the opponents hold COLLECTIVELY,
	#     = BANK_PER_RESOURCE - bank - my own hand. Bank and my hand are both
	#     public to me, so the bound is public; it says nothing about WHO holds
	#     what, which is the part that is actually hidden.
	#   - what I may GIVE, and whether I may accept, is read off my own hand.

	def _recv_is_legal(self, player, s):
		keeps_something = False
		for r in range(N_RESOURCES):
			mine = self.players[4*player, PA_RESOURCES + r]
			if TRADE_SETS[s, r] > BANK_PER_RESOURCE - self.globals_[0, GA_BANK + r] - mine:
				return False                    # nobody out there can supply that much
			if TRADE_SETS[s, r] == 0 and mine > 0:
				keeps_something = True
		# Guarantee the GIVE ply that follows will have at least one legal move:
		# the giveaway must be disjoint from the ask, so the player has to hold a
		# card of some type it is NOT asking for. Without this, an announcement
		# could walk into a phase with zero legal actions.
		return keeps_something

	def _give_is_legal(self, player, s):
		d = 4*player + 3
		for r in range(N_RESOURCES):
			if TRADE_SETS[s, r] > self.players[4*player, PA_RESOURCES + r]:
				return False
			if TRADE_SETS[s, r] > 0 and self.players[d, PD_TRADE_RECV + r] > 0:
				return False                    # never trade a resource against itself
		return True

	def _can_pay_recv_of(self, payer, proposer):
		# whoever accepts an offer hands over the proposer's PD_TRADE_RECV
		d = 4*proposer + 3
		for r in range(N_RESOURCES):
			if self.players[4*payer, PA_RESOURCES + r] < self.players[d, PD_TRADE_RECV + r]:
				return False
		return True

	def _robber_victims(self, h, player):
		# bit p set if player p can be robbed on hex h
		out = np.zeros(self.num_players, dtype=np.int8)
		for i in range(6):
			v = HEX_TO_VERTEX[h, i]
			o = self.vertices[v, V_OWNER]
			if o != 0 and o - 1 != player and self.players[4*(o - 1), PA_TOTAL_RES] > 0:
				out[o - 1] = 1
		return out

	def _setup_vertex(self, player):
		# The settlement whose road is still to be placed: the player's only
		# building with no incident road of its own. Derived rather than stored,
		# so that no board index ever sits in a global row (isometries).
		for v in range(N_VERTICES):
			if self.vertices[v, V_OWNER] != player + 1:
				continue
			has_road = False
			for k in range(3):
				if self.vertices[v, V_EDGE0 + k] == player + 1:
					has_road = True
			if not has_road:
				return v
		return 0   # no unroaded settlement: caller has a stale/inconsistent state

	def _next_actor(self):
		# Returns a CANONICAL-RELATIVE offset (0 = the player currently at
		# canonical index 0), never an absolute player id: make_move() applies its
		# auto-resolved moves in that frame and the caller rotates the board once,
		# afterwards.
		#   - setup road: it belongs to whoever owns the only settlement without
		#     a road (V_OWNER is canonical-relative), not necessarily to offset 0;
		#   - setup settlement: the snake order is absolute, so return how far it
		#     moves this step, not where it is.
		phase = self.globals_[0, GA_PHASE]
		if phase == PHASE_SETUP_ROAD:
			for v in range(N_VERTICES):
				if self.vertices[v, V_BUILDING] == 0:
					continue
				has_road = False
				for k in range(3):
					if self.vertices[v, V_EDGE0 + k] == self.vertices[v, V_OWNER]:
						has_road = True
				if not has_road:
					return int(self.vertices[v, V_OWNER]) - 1
			return 0   # unreachable in a consistent state; see _setup_vertex's own guard
		if phase == PHASE_SETUP_SETTLEMENT:
			s = int(self.globals_[1, GB_SETUP_STEP])
			if s == 0:
				return 0
			P = int(self.num_players)
			cur_abs = s if s < P else 2*P - 1 - s
			prev_abs = (s - 1) if (s - 1) < P else 2*P - 1 - (s - 1)
			return (cur_abs - prev_abs) % P
		if phase == PHASE_DISCARD:
			for p in range(self.num_players):
				if self.players[4*p + 2, PC_DISCARD_LEFT] > 0:
					return p
		# Trade: nothing is stored about whose turn it is to speak, it is read
		# back off the per-player statuses (see the BOARD DESCRIPTION header).
		if phase == PHASE_TRADE_OFFER:
			for p in range(self.num_players):
				if self.players[4*p + 3, PD_TRADE_STATUS] == TRADE_COMPOSING:
					return p
			return int(self.globals_[1, GB_TURN_PLAYER])   # unreachable when consistent
		if phase == PHASE_TRADE_ANSWER:
			t = int(self.globals_[1, GB_TURN_PLAYER])
			for i in range(1, self.num_players):
				p = (t + i) % self.num_players
				if self.players[4*p + 3, PD_TRADE_STATUS] == TRADE_NONE:
					return p
			return t                                        # unreachable when consistent
		# PHASE_TRADE_ACCEPT is answered by the turn player, which the line below
		# already returns.
		return int(self.globals_[1, GB_TURN_PLAYER])

	############################## VALID MOVES ################################

	def valid_moves(self, player):
		valids = np.zeros(N_ACTIONS, dtype=np.bool_)
		phase = self.globals_[0, GA_PHASE]

		if phase == PHASE_SETUP_SETTLEMENT:
			for v in range(N_VERTICES):
				if self._can_settle(v):
					valids[A_SETTLEMENT + v] = True

		elif phase == PHASE_SETUP_ROAD:
			v = self._setup_vertex(player)
			for k in range(3):
				e = VERTEX_TO_EDGE[v, k]
				if e != NO_EDGE and self._edge_owner(e) == 0:
					valids[A_ROAD + e] = True

		elif phase == PHASE_ROLL:
			valids[A_ROLL] = True
			if (self.players[4*player + 2, PC_DEV_PLAYED_THIS_TURN] == 0
					and self.players[4*player, PA_DEV_PLAYABLE + KNIGHT] > 0):
				valids[A_PLAY_DEV + 0] = True

		elif phase == PHASE_DISCARD:
			for r in range(N_RESOURCES):
				if self.players[4*player, PA_RESOURCES + r] > 0:
					valids[A_DISCARD + r] = True

		elif phase == PHASE_MOVE_ROBBER:
			for h in range(N_HEXES):
				if self.hexes[h, H_ROBBER] != 0:
					continue
				victims = self._robber_victims(h, player)
				if victims.sum() == 0:
					valids[A_ROBBER + h*int(self.num_players) + 0] = True
				else:
					for q in range(self.num_players):
						if victims[q] != 0:
							t = (q - player) % self.num_players
							valids[A_ROBBER + h*int(self.num_players) + t] = True

		elif phase == PHASE_ROAD_BUILDING:
			for e in range(N_EDGES):
				if self._edge_owner(e) == 0 and self._road_connected(e, player):
					valids[A_ROAD + e] = True

		elif phase == PHASE_TRADE_OFFER:
			# One phase, two plies: an empty PD_TRADE_RECV means the ask is still
			# to come, otherwise the offer is. No fourth phase id needed.
			asked = 0
			for r in range(N_RESOURCES):
				asked += self.players[4*player + 3, PD_TRADE_RECV + r]
			if asked == 0:
				for s in range(N_TRADE_SETS):
					if self._recv_is_legal(player, s):
						valids[A_TRADE_RECV + s] = True
			else:
				for s in range(N_TRADE_SETS):
					if self._give_is_legal(player, s):
						valids[A_TRADE_GIVE + s] = True

		elif phase == PHASE_TRADE_ANSWER:
			t = int(self.globals_[1, GB_TURN_PLAYER])
			valids[A_TRADE_NO] = True                       # always available
			if self._can_pay_recv_of(player, t):
				valids[A_TRADE_OK] = True
			if ENABLE_TRADE_COUNTER:                        # or counter-offer
				for s in range(N_TRADE_SETS):
					if self._recv_is_legal(player, s):
						valids[A_TRADE_RECV + s] = True

		elif phase == PHASE_TRADE_ACCEPT:
			valids[A_TRADE_ACCEPT + 0] = True               # refuse every counter
			for i in range(1, self.num_players):
				q = (player + i) % self.num_players
				if (self.players[4*q + 3, PD_TRADE_STATUS] == TRADE_OFFERED
						and self._can_pay_recv_of(player, q)):
					valids[A_TRADE_ACCEPT + i] = True

		elif phase == PHASE_MAIN:
			a, b, c = 4*player, 4*player + 1, 4*player + 2
			if self.players[b, PB_ROADS_LEFT] > 0 and self._can_afford(player, COST_ROAD):
				for e in range(N_EDGES):
					if self._edge_owner(e) == 0 and self._road_connected(e, player):
						valids[A_ROAD + e] = True
			if self.players[b, PB_SETTLEMENTS_LEFT] > 0 and self._can_afford(player, COST_SETTLEMENT):
				for v in range(N_VERTICES):
					if self._can_settle(v) and self._settlement_connected(v, player):
						valids[A_SETTLEMENT + v] = True
			if self.players[b, PB_CITIES_LEFT] > 0 and self._can_afford(player, COST_CITY):
				for v in range(N_VERTICES):
					if self.vertices[v, V_OWNER] == player + 1 and self.vertices[v, V_BUILDING] == 1:
						valids[A_CITY + v] = True
			if self.dev_deck_size() > 0 and self._can_afford(player, COST_DEV):
				valids[A_BUY_DEV] = True
			if self.players[c, PC_DEV_PLAYED_THIS_TURN] == 0:
				if self.players[a, PA_DEV_PLAYABLE + KNIGHT] > 0:
					valids[A_PLAY_DEV + 0] = True
				if (self.players[a, PA_DEV_PLAYABLE + ROAD_BUILDING] > 0
						and self.players[b, PB_ROADS_LEFT] > 0 and self._has_any_road(player)):
					valids[A_PLAY_DEV + 1] = True
				if self.players[a, PA_DEV_PLAYABLE + MONOPOLY] > 0:
					for r in range(N_RESOURCES):
						valids[A_MONOPOLY + r] = True
				if self.players[a, PA_DEV_PLAYABLE + YEAR_OF_PLENTY] > 0:
					for p in range(15):
						r1, r2 = YOP_PAIRS[p, 0], YOP_PAIRS[p, 1]
						need = 2 if r1 == r2 else 1
						if self.globals_[0, GA_BANK + r1] >= need and self.globals_[0, GA_BANK + r2] >= need:
							valids[A_YEAR_OF_PLENTY + p] = True
			if self.players[c, PC_TRADES_THIS_TURN] < MAX_TRADES_PER_TURN:
				for g in range(N_RESOURCES):
					ratio = self._trade_ratio(player, g)
					if self.players[a, PA_RESOURCES + g] < ratio:
						continue
					k = 0
					for get in range(N_RESOURCES):
						if get == g:
							continue
						if self.globals_[0, GA_BANK + get] > 0:
							valids[A_BANK_TRADE + g*4 + k] = True
						k += 1
			# Opening a player trade IS the A_TRADE_RECV action: no separate
			# "I would like to trade" id, which would cost a whole ply.
			if ENABLE_PLAYER_TRADE and self.globals_[1, GB_PLAYER_TRADE_DONE] == 0:
				for s in range(N_TRADE_SETS):
					if self._recv_is_legal(player, s):
						valids[A_TRADE_RECV + s] = True
			valids[A_END_TURN] = True

		return valids

	############################## MAKE MOVE ##################################

	def make_move(self, move, player, random_seed):
		self._apply(move, player, random_seed)
		# Never expose a choice with a single legal option: resolving it here
		# saves a whole MCTS search per occurrence, and they are frequent.
		for _guard in range(4 * MAX_ROUNDS):
			nxt = self._next_actor()
			if self.check_end_game(nxt).any():
				return nxt
			valids = self.valid_moves(nxt)
			n = 0
			only = 0
			for a in range(N_ACTIONS):
				if valids[a]:
					n += 1
					only = a
					if n > 1:
						break
			if n != 1:
				return nxt
			self._apply(only, nxt, random_seed)
		return self._next_actor()

	def _apply(self, move, player, random_seed):
		if A_ROAD <= move < A_ROAD + N_EDGES:
			self._do_road(move - A_ROAD, player, random_seed)
		elif A_SETTLEMENT <= move < A_SETTLEMENT + N_VERTICES:
			self._do_settlement(move - A_SETTLEMENT, player, random_seed)
		elif A_CITY <= move < A_CITY + N_VERTICES:
			self._do_city(move - A_CITY, player)
		elif move == A_BUY_DEV:
			self._pay(player, COST_DEV)
			self._draw_dev(player, random_seed)
		elif move == A_PLAY_DEV + 0:
			self.players[4*player, PA_DEV_PLAYABLE + KNIGHT] -= 1
			self.globals_[1, GB_DEV_PLAYED + KNIGHT] += 1
			self.players[4*player + 1, PB_KNIGHTS] += 1
			self.players[4*player + 2, PC_DEV_PLAYED_THIS_TURN] = 1
			self._refresh_totals(player)
			self._update_largest_army(player)
			self.globals_[0, GA_PHASE] = PHASE_MOVE_ROBBER
		elif move == A_PLAY_DEV + 1:
			self.players[4*player, PA_DEV_PLAYABLE + ROAD_BUILDING] -= 1
			self.globals_[1, GB_DEV_PLAYED + ROAD_BUILDING] += 1
			self.players[4*player + 2, PC_DEV_PLAYED_THIS_TURN] = 1
			self._refresh_totals(player)
			self.globals_[1, GB_PENDING_COUNT] = 2
			self.globals_[0, GA_PHASE] = PHASE_ROAD_BUILDING
		elif A_ROBBER <= move < A_ROBBER + N_HEXES * int(self.num_players):
			h, t = divmod(move - A_ROBBER, self.num_players)
			self._do_robber(h, t, player, random_seed)
		elif move == A_ROLL:
			self._roll_and_produce(player, random_seed)
		elif A_MONOPOLY <= move < A_MONOPOLY + N_RESOURCES:
			self._do_monopoly(move - A_MONOPOLY, player)
		elif A_YEAR_OF_PLENTY <= move < A_YEAR_OF_PLENTY + 15:
			self._do_year_of_plenty(move - A_YEAR_OF_PLENTY, player)
		elif A_BANK_TRADE <= move < A_BANK_TRADE + 20:
			self._do_bank_trade(move - A_BANK_TRADE, player)
		elif A_DISCARD <= move < A_DISCARD + N_RESOURCES:
			self._do_discard(move - A_DISCARD, player)
		elif A_TRADE_RECV <= move < A_TRADE_RECV + N_TRADE_SETS:
			self._do_trade_recv(move - A_TRADE_RECV, player)
		elif A_TRADE_GIVE <= move < A_TRADE_GIVE + N_TRADE_SETS:
			self._do_trade_give(move - A_TRADE_GIVE, player)
		elif move == A_TRADE_OK:
			self._do_trade_ok(player)
		elif move == A_TRADE_NO:
			self._do_trade_no(player)
		elif A_TRADE_ACCEPT <= move < A_TRADE_ACCEPT + int(self.num_players):
			self._do_trade_accept(move - A_TRADE_ACCEPT, player)
		elif move == A_END_TURN:
			self._end_turn(player)

	############################## MAKE MOVE: PRIMITIVES ######################

	def _next_counter(self):
		c = self.globals_[1, GB_CHANCE_COUNTER]
		self.globals_[1, GB_CHANCE_COUNTER] = (c + 1) % 100
		return np.int64(self.get_round()) * 100 + np.int64(c)

	def _draw(self, random_seed, counter, m):
		if random_seed == 0:
			return np.int64(np.random.randint(0, m))
		return np.int64(hashed_draw(random_seed, counter, m))

	def _pay(self, player, cost):
		for r in range(N_RESOURCES):
			self.players[4*player, PA_RESOURCES + r] -= cost[r]
			self.globals_[0, GA_BANK + r] += cost[r]
		self._refresh_totals(player)

	def _refresh_totals(self, player):
		tot = 0
		for r in range(N_RESOURCES):
			tot += self.players[4*player, PA_RESOURCES + r]
		self.players[4*player, PA_TOTAL_RES] = tot
		nb_new = 0
		dev = 0
		for k in range(N_DEV_TYPES):
			nb_new += self.players[4*player + 1, PB_DEV_NEW + k]
			dev += self.players[4*player, PA_DEV_PLAYABLE + k]
		self.players[4*player, PA_TOTAL_DEV] = dev + nb_new
		self.players[4*player + 2, PC_TOTAL_DEV_NEW] = nb_new
		self.players[4*player + 2, PC_VP_DEV] = (self.players[4*player, PA_DEV_PLAYABLE + VICTORY_POINT]
		                                         + self.players[4*player + 1, PB_DEV_NEW + VICTORY_POINT])

	def _refresh_vp(self, player):
		s = 0
		c = 0
		for v in range(N_VERTICES):
			if self.vertices[v, V_OWNER] == player + 1:
				if self.vertices[v, V_BUILDING] == 1:
					s += 1
				elif self.vertices[v, V_BUILDING] == 2:
					c += 1
		self.players[4*player + 2, PC_VP_PUBLIC] = (s + 2*c
		                                            + 2*self.players[4*player + 1, PB_HAS_ROAD]
		                                            + 2*self.players[4*player + 1, PB_HAS_ARMY])

	def _do_road(self, e, player, random_seed):
		self._set_edge(e, player + 1)
		self.players[4*player + 1, PB_ROADS_LEFT] -= 1
		phase = self.globals_[0, GA_PHASE]
		if phase == PHASE_SETUP_ROAD:
			self._advance_setup(player, random_seed)
		elif phase == PHASE_ROAD_BUILDING:
			self.globals_[1, GB_PENDING_COUNT] -= 1
			if self.globals_[1, GB_PENDING_COUNT] == 0 or not self._has_any_road(player) \
					or self.players[4*player + 1, PB_ROADS_LEFT] == 0:
				self.globals_[1, GB_PENDING_COUNT] = 0
				self.globals_[0, GA_PHASE] = PHASE_MAIN
		else:
			self._pay(player, COST_ROAD)
		self._update_longest_road()

	def _do_settlement(self, v, player, random_seed):
		self.vertices[v, V_BUILDING] = 1
		self.vertices[v, V_OWNER] = player + 1
		self.players[4*player + 1, PB_SETTLEMENTS_LEFT] -= 1
		if self.vertices[v, V_PORT] != PORT_NONE:
			self.players[4*player + 2, PC_PORTS + self.vertices[v, V_PORT] - 1] = 1
		if self.globals_[0, GA_PHASE] == PHASE_SETUP_SETTLEMENT:
			# second settlement of the snake: it produces immediately
			if self.globals_[1, GB_SETUP_STEP] >= self.num_players:
				for k in range(3):
					h = VERTEX_TO_HEX[v, k]
					if h != NO_HEX and self.hexes[h, H_TYPE] != HEX_DESERT:
						self._gain(player, HEX_TO_RESOURCE[self.hexes[h, H_TYPE]], 1)
			self.globals_[0, GA_PHASE] = PHASE_SETUP_ROAD
		else:
			self._pay(player, COST_SETTLEMENT)
		self._refresh_vp(player)
		self._update_longest_road()

	def _do_city(self, v, player):
		self.vertices[v, V_BUILDING] = 2
		self.players[4*player + 1, PB_SETTLEMENTS_LEFT] += 1
		self.players[4*player + 1, PB_CITIES_LEFT] -= 1
		self._pay(player, COST_CITY)
		self._refresh_vp(player)

	def _gain(self, player, r, n):
		take = min(n, self.globals_[0, GA_BANK + r])
		self.globals_[0, GA_BANK + r] -= take
		self.players[4*player, PA_RESOURCES + r] += take
		self._refresh_totals(player)

	def _advance_setup(self, player, random_seed):
		self.globals_[1, GB_SETUP_STEP] += 1
		if self.globals_[1, GB_SETUP_STEP] >= 2*self.num_players:
			self.globals_[1, GB_TURN_PLAYER] = 0
			self._start_turn(0)
		else:
			self.globals_[0, GA_PHASE] = PHASE_SETUP_SETTLEMENT

	def _start_turn(self, player):
		for k in range(N_DEV_TYPES):
			self.players[4*player, PA_DEV_PLAYABLE + k] += self.players[4*player + 1, PB_DEV_NEW + k]
			self.players[4*player + 1, PB_DEV_NEW + k] = 0
		self.players[4*player + 2, PC_DEV_PLAYED_THIS_TURN] = 0
		self.players[4*player + 2, PC_TRADES_THIS_TURN] = 0
		self.globals_[1, GB_PLAYER_TRADE_DONE] = 0
		self._refresh_totals(player)
		self.globals_[0, GA_PHASE] = PHASE_ROLL

	def _end_turn(self, player):
		self._set_round(self.get_round() + 1)
		nxt = (self.globals_[1, GB_TURN_PLAYER] + 1) % self.num_players
		self.globals_[1, GB_TURN_PLAYER] = nxt
		self._start_turn(nxt)

	def _roll_and_produce(self, player, random_seed):
		c = self._next_counter()
		dice = 2 + self._draw(random_seed, 2*c, 6) + self._draw(random_seed, 2*c + 1, 6)
		self.globals_[0, GA_DICE] = dice
		if dice == 7:
			total = 0
			for p in range(self.num_players):
				n = self.players[4*p, PA_TOTAL_RES]
				self.players[4*p + 2, PC_DISCARD_LEFT] = n // 2 if n > HAND_LIMIT_ON_SEVEN else 0
				total += self.players[4*p + 2, PC_DISCARD_LEFT]
			self.globals_[0, GA_PHASE] = PHASE_DISCARD if total > 0 else PHASE_MOVE_ROBBER
			return
		# production, with the official bank-shortage rule
		owed = np.zeros((self.num_players, N_RESOURCES), dtype=np.int8)
		for h in range(N_HEXES):
			if self.hexes[h, H_ROBBER] != 0 or self.hexes[h, H_TYPE] == HEX_DESERT:
				continue
			if TOKEN_VALUES[self.hexes[h, H_TOKEN]] != dice:
				continue
			r = HEX_TO_RESOURCE[self.hexes[h, H_TYPE]]
			for i in range(6):
				v = HEX_TO_VERTEX[h, i]
				o = self.vertices[v, V_OWNER]
				if o != 0:
					owed[o - 1, r] += self.vertices[v, V_BUILDING]
		for r in range(N_RESOURCES):
			claimants = 0
			total = 0
			for p in range(self.num_players):
				if owed[p, r] > 0:
					claimants += 1
					total += owed[p, r]
			if total == 0:
				continue
			# if the bank runs short and several players are owed, nobody gets any
			if total > self.globals_[0, GA_BANK + r] and claimants > 1:
				continue
			for p in range(self.num_players):
				if owed[p, r] > 0:
					self._gain(p, r, owed[p, r])
		self.globals_[0, GA_PHASE] = PHASE_MAIN

	def _do_discard(self, r, player):
		self.players[4*player, PA_RESOURCES + r] -= 1
		self.globals_[0, GA_BANK + r] += 1
		self.players[4*player + 2, PC_DISCARD_LEFT] -= 1
		self._refresh_totals(player)
		remaining = 0
		for p in range(self.num_players):
			remaining += self.players[4*p + 2, PC_DISCARD_LEFT]
		if remaining == 0:
			self.globals_[0, GA_PHASE] = PHASE_MOVE_ROBBER

	def _do_robber(self, h, t, player, random_seed):
		for i in range(N_HEXES):
			self.hexes[i, H_ROBBER] = 0
		self.hexes[h, H_ROBBER] = 1
		if t != 0:
			victim = (player + t) % self.num_players
			c = self._next_counter()
			pick = self._draw(random_seed, c, self.players[4*victim, PA_TOTAL_RES])
			acc = 0
			for r in range(N_RESOURCES):
				acc += self.players[4*victim, PA_RESOURCES + r]
				if pick < acc:
					self.players[4*victim, PA_RESOURCES + r] -= 1
					self.players[4*player, PA_RESOURCES + r] += 1
					break
			self._refresh_totals(victim)
			self._refresh_totals(player)
		self.globals_[0, GA_PHASE] = PHASE_MAIN

	def _do_monopoly(self, r, player):
		self.players[4*player, PA_DEV_PLAYABLE + MONOPOLY] -= 1
		self.globals_[1, GB_DEV_PLAYED + MONOPOLY] += 1
		self.players[4*player + 2, PC_DEV_PLAYED_THIS_TURN] = 1
		for p in range(self.num_players):
			if p == player:
				continue
			n = self.players[4*p, PA_RESOURCES + r]
			self.players[4*p, PA_RESOURCES + r] = 0
			self.players[4*player, PA_RESOURCES + r] += n
			self._refresh_totals(p)
		self._refresh_totals(player)

	def _do_year_of_plenty(self, p, player):
		self.players[4*player, PA_DEV_PLAYABLE + YEAR_OF_PLENTY] -= 1
		self.globals_[1, GB_DEV_PLAYED + YEAR_OF_PLENTY] += 1
		self.players[4*player + 2, PC_DEV_PLAYED_THIS_TURN] = 1
		self._gain(player, YOP_PAIRS[p, 0], 1)
		self._gain(player, YOP_PAIRS[p, 1], 1)

	def _do_bank_trade(self, code, player):
		give, k = divmod(code, 4)
		get = k if k < give else k + 1
		ratio = self._trade_ratio(player, give)
		self.players[4*player, PA_RESOURCES + give] -= ratio
		self.globals_[0, GA_BANK + give] += ratio
		self.globals_[0, GA_BANK + get] -= 1
		self.players[4*player, PA_RESOURCES + get] += 1
		self.players[4*player + 2, PC_TRADES_THIS_TURN] += 1
		self._refresh_totals(player)

	############################## MAKE MOVE: PLAYER TRADE ####################
	# See the ACTION DESCRIPTION header for the protocol and its ply bound. Every
	# transition below is driven by the per-player statuses alone, so there is no
	# protocol counter that swap_players could leave stale.

	def _do_trade_recv(self, s, player):
		d = 4*player + 3
		for r in range(N_RESOURCES):
			self.players[d, PD_TRADE_RECV + r] = TRADE_SETS[s, r]
			self.players[d, PD_TRADE_GIVE + r] = 0
		self.players[d, PD_TRADE_STATUS] = TRADE_COMPOSING
		self.globals_[0, GA_PHASE] = PHASE_TRADE_OFFER

	def _do_trade_give(self, s, player):
		d = 4*player + 3
		for r in range(N_RESOURCES):
			self.players[d, PD_TRADE_GIVE + r] = TRADE_SETS[s, r]
		self.players[d, PD_TRADE_STATUS] = TRADE_OFFERED
		self._advance_trade()

	def _do_trade_no(self, player):
		self.players[4*player + 3, PD_TRADE_STATUS] = TRADE_REFUSED
		self._advance_trade()

	def _do_trade_ok(self, player):
		# a responder accepts the TURN PLAYER's standing offer: that closes the
		# round table at once, whatever is still stacked behind it
		self._execute_trade(int(self.globals_[1, GB_TURN_PLAYER]), player)
		self._end_trade()

	def _do_trade_accept(self, t, player):
		# `player` is the turn player picking among the stacked counters;
		# t is a RELATIVE id, 0 meaning "refuse them all"
		if t != 0:
			self._execute_trade((player + t) % self.num_players, player)
		self._end_trade()

	def _execute_trade(self, proposer, accepter):
		# The proposer receives its RECV and parts with its GIVE; the accepter does
		# the reverse. Both sides were checked when they committed -- the proposer's
		# GIVE by _give_is_legal, the accepter's side by _can_pay_recv_of -- and no
		# action in between can spend a card, since the only moves available are
		# answers to this same announcement.
		d = 4*proposer + 3
		for r in range(N_RESOURCES):
			n_recv = self.players[d, PD_TRADE_RECV + r]
			n_give = self.players[d, PD_TRADE_GIVE + r]
			self.players[4*accepter, PA_RESOURCES + r] += n_give - n_recv
			self.players[4*proposer, PA_RESOURCES + r] += n_recv - n_give
		self._refresh_totals(accepter)
		self._refresh_totals(proposer)

	def _advance_trade(self):
		t = int(self.globals_[1, GB_TURN_PLAYER])
		for i in range(1, self.num_players):            # anyone still to answer?
			p = (t + i) % self.num_players
			if self.players[4*p + 3, PD_TRADE_STATUS] == TRADE_NONE:
				self.globals_[0, GA_PHASE] = PHASE_TRADE_ANSWER
				return
		for i in range(1, self.num_players):            # any counter left standing?
			p = (t + i) % self.num_players
			if self.players[4*p + 3, PD_TRADE_STATUS] == TRADE_OFFERED:
				self.globals_[0, GA_PHASE] = PHASE_TRADE_ACCEPT
				return
		self._end_trade()                               # everyone refused

	def _end_trade(self):
		for p in range(self.num_players):
			d = 4*p + 3
			for r in range(N_RESOURCES):
				self.players[d, PD_TRADE_RECV + r] = 0
				self.players[d, PD_TRADE_GIVE + r] = 0
			self.players[d, PD_TRADE_STATUS] = TRADE_NONE
		self.globals_[1, GB_PLAYER_TRADE_DONE] = 1      # spent even when nobody accepted
		self.globals_[0, GA_PHASE] = PHASE_MAIN

	def _draw_dev(self, player, random_seed):
		total = 0
		for k in range(N_DEV_TYPES):
			total += self.globals_[0, GA_DEV_DECK + k]
		c = self._next_counter()
		pick = self._draw(random_seed, c, total)
		acc = 0
		for k in range(N_DEV_TYPES):
			acc += self.globals_[0, GA_DEV_DECK + k]
			if pick < acc:
				self.globals_[0, GA_DEV_DECK + k] -= 1
				self.players[4*player + 1, PB_DEV_NEW + k] += 1
				break
		self._refresh_totals(player)
		self._refresh_vp(player)

	############################## BONUS CARDS ################################

	def _update_largest_army(self, player):
		k = self.players[4*player + 1, PB_KNIGHTS]
		if k < MIN_KNIGHTS_FOR_ARMY:
			return
		holder = -1
		for p in range(self.num_players):
			if self.players[4*p + 1, PB_HAS_ARMY] != 0:
				holder = p
		if holder == player:
			return
		if holder >= 0 and k <= self.players[4*holder + 1, PB_KNIGHTS]:
			return
		if holder >= 0:
			self.players[4*holder + 1, PB_HAS_ARMY] = 0
			self._refresh_vp(holder)
		self.players[4*player + 1, PB_HAS_ARMY] = 1
		self._refresh_vp(player)

	def _update_longest_road(self):
		best_len = 0
		nb_best = 0
		best_p = -1
		holder = -1
		for p in range(self.num_players):
			l = longest_road(self.vertices, p + 1)
			self.players[4*p + 1, PB_ROAD_LENGTH] = l
			if self.players[4*p + 1, PB_HAS_ROAD] != 0:
				holder = p
		for p in range(self.num_players):
			l = self.players[4*p + 1, PB_ROAD_LENGTH]
			if l > best_len:
				best_len, nb_best, best_p = l, 1, p
			elif l == best_len:
				nb_best += 1
				best_p = -1                     # tied: no single claimant

		if best_len < MIN_LENGTH_FOR_ROAD:
			if holder >= 0:
				self.players[4*holder + 1, PB_HAS_ROAD] = 0
				self._refresh_vp(holder)
			return
		if holder >= 0 and self.players[4*holder + 1, PB_ROAD_LENGTH] == best_len:
			return                              # the incumbent keeps it, ties included
		# the incumbent is no longer the longest: it loses the card, and nobody
		# takes it if the new best is tied
		if holder >= 0:
			self.players[4*holder + 1, PB_HAS_ROAD] = 0
			self._refresh_vp(holder)
		if best_p >= 0:
			self.players[4*best_p + 1, PB_HAS_ROAD] = 1
			self._refresh_vp(best_p)
