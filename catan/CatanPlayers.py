import numpy as np
import random

from .CatanDisplay import move_to_str, print_board
from .CatanConstants import *


class RandomPlayer():
	def __init__(self, game):
		self.game = game

	def play(self, board, nb_moves):
		valids = self.game.getValidMoves(board, player=0)
		action = random.choices(range(self.game.getActionSize()), weights=valids.astype(np.int16), k=1)[0]
		return action


class HumanPlayer():
	def __init__(self, game):
		self.game = game

	def show_all_moves(self, valid):
		for i, v in enumerate(valid):
			if v:
				print(f'{i} = {move_to_str(i, 0, short=True)}', end='   ')
		print()

	def show_main_moves(self, valid):
		# Roads and settlements are the two blocks that flood the screen: show a
		# count and the target ids only, everything else in full.
		print()
		for name, start, size in (('roads', A_ROAD, N_EDGES),
		                          ('settlements', A_SETTLEMENT, N_VERTICES),
		                          ('cities', A_CITY, N_VERTICES),
		                          ('trade asks', A_TRADE_RECV, N_TRADE_SETS),
		                          ('trade offers', A_TRADE_GIVE, N_TRADE_SETS)):
			ids = [i - start for i in range(start, start + size) if valid[i]]
			if ids:
				print(f'{name}: ' + ' '.join(f'{start + i}({move_to_str(start + i, 0, short=True)})' for i in ids))
		for i in list(range(A_BUY_DEV, N_ACTIONS_V1)) + [A_TRADE_OK, A_TRADE_NO] \
				+ list(range(A_TRADE_ACCEPT, A_TRADE_ACCEPT + N_PLAYERS)):
			if valid[i]:
				print(f'{i} = {move_to_str(i, 0, short=True)}', end='   ')
		print('\n(+ to show all moves)')

	def play(self, board, nb_moves):
		valid = self.game.getValidMoves(board, 0)
		self.show_main_moves(valid)
		while True:
			input_move = input()
			if input_move == '+':
				self.show_all_moves(valid)
			else:
				try:
					a = int(input_move)
					if not valid[a]:
						raise Exception('')
					break
				except Exception:
					print('Invalid move:', input_move)
		return a


############################# HEURISTIC BASELINE ##############################
#
# A ONE-PLY GREEDY DOES NOT WORK HERE, and the failure is instructive: ranked on
# (victory points, production, hand size) it played 2 settlements, 4 roads, then
# ended its turn 84 times and discarded 200 cards to sevens. Every build lowers
# the hand and no road raises VP or production, so "do nothing" always won the
# tie-break. Catan punishes hoarding, which one ply cannot see.
# What follows is therefore a rule-based player, not a search: a fixed priority
# list over the phases. It is still weak -- no lookahead, blind to blocking, to
# card counting and to what opponents are about to build -- but it is a baseline
# a trained network should beat comfortably, which a random player is not.


def vertex_value(state, v):
	"""Expected production of a settlement on v, in pips (36*P per turn)."""
	total = 0
	for k in range(3):
		h = VERTEX_TO_HEX[v, k]
		if h != NO_HEX:
			total += int(state[N_VERTICES + h, H_PIPS])
	if state[v, V_PORT] != PORT_NONE:
		total += 1                      # a port is worth roughly one pip
	return total


def can_settle(state, v):
	if state[v, V_BUILDING] != 0:
		return False
	for k in range(3):
		o = VERTEX_TO_VERTEX[v, k]
		if o != NO_VERTEX and state[o, V_BUILDING] != 0:
			return False
	return True


def production_pips(state, player):
	"""Expected production of `player` over the whole board, in pips."""
	total = 0
	for v in range(N_VERTICES):
		if state[v, V_OWNER] == player + 1:
			for k in range(3):
				h = VERTEX_TO_HEX[v, k]
				if h != NO_HEX:
					total += int(state[v, V_BUILDING]) * int(state[N_VERTICES + h, H_PIPS])
	return total


class GreedyPlayer():
	"""Rule-based baseline. See the comment above for why it is not a greedy."""

	def __init__(self, game):
		self.game = game

	def play(self, board, nb_moves):
		valids = self.game.getValidMoves(board, 0)
		row_a = N_VERTICES + N_HEXES
		row_global = row_a + ROWS_PER_PLAYER * self.game.num_players
		phase = board[row_global, GA_PHASE]

		def pick(candidates):
			best = max(s for s, _ in candidates)
			return random.choice([m for s, m in candidates if s == best])

		# --- placements ---------------------------------------------------
		if phase == PHASE_SETUP_SETTLEMENT:
			return pick([(vertex_value(board, m - A_SETTLEMENT), int(m))
			             for m in np.flatnonzero(valids)])
		if phase in (PHASE_SETUP_ROAD, PHASE_ROAD_BUILDING):
			return pick([(self._road_score(board, int(m) - A_ROAD), int(m))
			             for m in np.flatnonzero(valids)])

		# --- player trade ----------------------------------------------------
		# This baseline never OPENS a trade (A_TRADE_RECV is simply absent from
		# the main-phase priority list below), but the other players can drag it
		# into the answer phases, and falling through to that list would return
		# A_END_TURN -- illegal there, and an assert in Arena rather than a loss.
		# Accepting whenever it nets more cards than it gives is a deliberately
		# crude placeholder: it ignores WHICH resources move, so a trained net
		# should exploit it easily. It is a control, not an opponent.
		if phase == PHASE_TRADE_ANSWER:
			turn_player = int(board[row_global + 1, GB_TURN_PLAYER])
			d = row_a + ROWS_PER_PLAYER * turn_player + 3
			recv = board[d, PD_TRADE_RECV:PD_TRADE_RECV + N_RESOURCES].sum()
			give = board[d, PD_TRADE_GIVE:PD_TRADE_GIVE + N_RESOURCES].sum()
			if valids[A_TRADE_OK] and give > recv:
				return A_TRADE_OK
			return A_TRADE_NO
		if phase == PHASE_TRADE_ACCEPT:
			return A_TRADE_ACCEPT + 0           # refuse every counter-offer
		if phase == PHASE_TRADE_OFFER:
			# unreachable while this baseline never announces, but a wrong guess
			# here would be an illegal move rather than a bad one: pick a legal id
			return int(np.flatnonzero(valids)[0])

		# --- forced-ish phases ---------------------------------------------
		if phase == PHASE_ROLL:
			return A_ROLL                       # never spend a knight before rolling
		if phase == PHASE_DISCARD:
			return pick([(int(board[row_a, PA_RESOURCES + int(m) - A_DISCARD]), int(m))
			             for m in np.flatnonzero(valids)])
		if phase == PHASE_MOVE_ROBBER:
			return pick([(self._robber_score(board, int(m)), int(m))
			             for m in np.flatnonzero(valids)])

		# --- main phase: a priority list ------------------------------------
		hand = int(board[row_a, PA_TOTAL_RES])
		cities = [int(m) for m in np.flatnonzero(valids) if A_CITY <= m < A_CITY + N_VERTICES]
		if cities:
			return pick([(production_pips_gain(board, m - A_CITY), m) for m in cities])
		settle = [int(m) for m in np.flatnonzero(valids) if A_SETTLEMENT <= m < A_SETTLEMENT + N_VERTICES]
		if settle:
			return pick([(vertex_value(board, m - A_SETTLEMENT), m) for m in settle])
		roads = [int(m) for m in np.flatnonzero(valids) if A_ROAD <= m < A_ROAD + N_EDGES]
		if roads:
			scored = [(self._road_score(board, m - A_ROAD), m) for m in roads]
			if max(s for s, _ in scored) > 0:    # only if it opens a spot worth taking
				return pick(scored)
		if valids[A_BUY_DEV] and hand >= 6:      # spend rather than discard on a seven
			return A_BUY_DEV
		trades = [int(m) for m in np.flatnonzero(valids) if A_BANK_TRADE <= m < A_BANK_TRADE + 20]
		if trades and hand > HAND_LIMIT_ON_SEVEN:
			return pick([(self._trade_score(board, m - A_BANK_TRADE), m) for m in trades])
		if valids[A_BUY_DEV]:
			return A_BUY_DEV
		return A_END_TURN

	def _road_score(self, board, e):
		"""Best settle-able spot this road would put within reach."""
		best = 0
		for i in range(2):
			v = EDGE_TO_VERTEX[e, i]
			for cand in [v] + [o for o in VERTEX_TO_VERTEX[v] if o != NO_VERTEX]:
				if can_settle(board, cand):
					best = max(best, vertex_value(board, cand))
		return best

	def _robber_score(self, board, move):
		h, t = divmod(move - A_ROBBER, self.game.num_players)
		score = 0
		for i in range(6):
			v = HEX_TO_VERTEX[h, i]
			o = board[v, V_OWNER]
			if o == 1:
				score -= 10 * int(board[v, V_BUILDING])     # never block ourselves
			elif o != 0:
				score += int(board[v, V_BUILDING]) * int(board[N_VERTICES + h, H_PIPS])
		if t != 0:
			score += int(board[N_VERTICES + N_HEXES + 4 * t, PA_TOTAL_RES])
		return score

	def _trade_score(self, board, code):
		give, k = divmod(code, 4)
		get = k if k < give else k + 1
		row_a = N_VERTICES + N_HEXES
		# trade the most abundant for the rarest of the settlement cost
		return int(board[row_a, PA_RESOURCES + give]) - int(board[row_a, PA_RESOURCES + get]) \
			+ (2 if COST_SETTLEMENT[get] > 0 else 0)


def production_pips_gain(state, v):
	"""Pips gained by upgrading the settlement on v to a city."""
	return vertex_value(state, v)
