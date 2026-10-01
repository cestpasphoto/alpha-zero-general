#!/usr/bin/env python3
"""
CatanFrameTest.py -- the game must not depend on the frame it is stepped in.

make_move() is called on CANONICAL boards by MCTS (the mover is index 0) and on
ABSOLUTE boards by Coach, Arena and pit (the mover is any seat). Both must play
exactly the same game. They did not: the setup order came out 0,1,1,0,2,2 on
absolute boards (0,1,2,2,1,0 in MCTS), and discards after a 7 were resolved in
index order, i.e. in a frame-dependent order.

Whole random games are stepped twice in lockstep, with the same actions and the
same chance seed: once on the absolute board, once on the canonical board of the
mover (then rotated back). Board and next player must match at every ply, in
every phase (setup, trade, 7, discards, robber...).

Usage (repository root):   python -m catan.CatanFrameTest [n_games]
   or (from catan/):        python CatanFrameTest.py [n_games]
Exit code 1 on the first divergence, whose state is saved to frame_divergence.npz.
"""
import sys

import numpy as np
from numba import njit

try:
	from .CatanGame import CatanGame
	from .CatanConstants import N_PLAYERS, ROW_GLOBAL, GA_PHASE, PHASE_SETUP_SETTLEMENT, A_TRADE_RECV, A_TRADE_ACCEPT
except ImportError:
	import os
	sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from catan.CatanGame import CatanGame
	from catan.CatanConstants import N_PLAYERS, ROW_GLOBAL, GA_PHASE, PHASE_SETUP_SETTLEMENT, A_TRADE_RECV, A_TRADE_ACCEPT


@njit(cache=True)
def _seed_numba(s):
	np.random.seed(s)   # init_game shuffles the board with numba's own RNG


def run(n_games, seed0=100, max_plies=3000, trade_bias=0.3):
	g, P = CatanGame(), N_PLAYERS
	plies, setup_orders = 0, set()
	for gi in range(n_games):
		_seed_numba(seed0 + gi)
		rng = np.random.default_rng(seed0 + gi)
		b, cur, order = np.copy(g.getInitBoard()), 0, []
		for ply in range(max_plies):
			cb = np.copy(g.getCanonicalForm(b, cur))
			if int(cb[ROW_GLOBAL, GA_PHASE]) == PHASE_SETUP_SETTLEMENT:
				order.append(cur)
			valid = np.flatnonzero(g.getValidMoves(cb, 0))
			trade = [a for a in valid if A_TRADE_RECV <= a < A_TRADE_ACCEPT + P]
			a = int(rng.choice(trade)) if trade and rng.random() < trade_bias else int(rng.choice(valid))
			seed = 1 + ply   # non-zero: deterministic chance, identical in both frames
			ba, na = g.getNextState(b, cur, a, random_seed=seed)
			ba, na = np.copy(ba), int(na)
			bc, rel = g.getNextState(cb, 0, a, random_seed=seed)
			bc = np.copy(g.getCanonicalForm(np.copy(bc), (P - cur) % P))
			nc = (cur + int(rel)) % P
			plies += 1
			if not (np.array_equal(ba, bc) and na == nc):
				np.savez('frame_divergence.npz', board=b, cur=cur, action=a, seed=seed)
				print(f'[FAIL] game {gi} ply {ply}: phase {int(cb[ROW_GLOBAL, GA_PHASE])}, mover {cur}, action {a}: '
				      f'next player {na} (absolute) vs {nc} (canonical), boards '
				      f'{"equal" if np.array_equal(ba, bc) else "DIFFER"}; state saved to frame_divergence.npz')
				return False
			b, cur = ba, na
			if np.any(g.getGameEnded(b, cur)):
				break
		setup_orders.add(tuple(order))
	expected = tuple(list(range(P)) + list(range(P - 1, -1, -1)))
	if setup_orders != {expected}:
		print(f'[FAIL] setup order {sorted(setup_orders)}, expected {expected}')
		return False
	print(f'[OK] {n_games} games, {plies} plies: absolute and canonical stepping identical; setup order {expected}')
	return True


if __name__ == '__main__':
	sys.exit(0 if run(int(sys.argv[1]) if len(sys.argv) > 1 else 50) else 1)
