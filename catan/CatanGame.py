import sys
sys.path.append('..')
from Game import Game
from .CatanConstants import N_PLAYERS as NUMBER_PLAYERS
from .CatanConstants import (N_ISOMETRIES, SYM_TRIVIAL_MAX_LEGAL, N_SYM_TRIVIAL, N_PHASES, N_TRADE_SETS,
                             TRADE_SETS, ROW_GLOBAL, ROW_PLAYER, ROWS_PER_PLAYER, GA_PHASE, GB_TURN_PLAYER,
                             PD_TRADE_RECV, PD_TRADE_GIVE, N_RESOURCES, PHASE_MAIN, PHASE_TRADE_OFFER,
                             PHASE_TRADE_ANSWER, PHASE_TRADE_ACCEPT, A_TRADE_RECV, A_TRADE_GIVE, A_TRADE_OK,
                             A_TRADE_NO, A_TRADE_ACCEPT, GB_ROUND_LO, GB_ROUND_HI, PA_TOTAL_RES,
                             MAX_ROUNDS, ACTION_BLOCK_OF, N_BLOCKS)
from .CatanLogicNumba import Board, observation_size, action_size
from .CatanDisplay import move_to_str, print_board
import numpy as np
import os
import threading

# Catan games run to ~300-400 decisions on average and up to 4*MAX_ROUNDS=1600
# in the worst case (the same bound make_move()'s auto-resolve guard and
# CatanTest.py's playout loop use) -- 5 to 10x longer than this framework's
# other games. MCTS.search() is recursive (one Python stack frame per ply), so
# a deep, narrow tree -- expected under an early, undertrained network whose
# priors concentrate PUCT on a single line -- can exceed Python's default
# 1000-frame limit well before reaching a real game end, raising a bare
# RecursionError. Bumping the limit here (12x the 1600 worst case, generous
# headroom) runs at Catan's own import time, before Coach spawns any self-play
# threads, and does not affect the other games.
# This raises Python's own counter, not the underlying OS stack: it was tested
# safe to a simulated depth of 25000 with this limit on Linux (8MB thread
# stack, the common default). If a HARD crash (segfault, not a catchable
# RecursionError) ever appears instead, the OS stack itself is too small for
# this platform/thread and needs raising independently, e.g. `ulimit -s
# 65532` before launching python (macOS/Linux), or threading.stack_size() set
# before the offending thread is created.
if sys.getrecursionlimit() < 20000:
	sys.setrecursionlimit(20000)


class _TradeStats:
	"""Per-process counters of what the MCTS policy puts on the trade actions,
	written to <dir>/trade_stats_<pid>.npz every `flush_every` positions and at
	exit (self-play runs in several processes; CatanTradeStats.py sums the
	files). Everything is policy MASS, i.e. the MCTS visit distribution at the
	root, not the sampled move: it is what the search wants, before the
	temperature."""

	def __init__(self, out_dir, flush_every=2000):
		self.out_dir, self.flush_every, self.n = out_dir, flush_every, 0
		os.makedirs(out_dir, exist_ok=True)
		self.path = os.path.join(out_dir, f'trade_stats_{os.getpid()}.npz')
		self.set_index = {tuple(int(x) for x in TRADE_SETS[i]): i for i in range(N_TRADE_SETS)}
		self.n_pos = np.zeros(N_PHASES, np.int64)          # positions seen, per phase
		self.main = np.zeros(5, np.float64)                # MAIN, when opening is legal: [positions, pi mass on opening, argmax is opening, legal opening ids, legal moves]
		self.legal = np.zeros(N_PHASES, np.float64)        # sum of the legal-move count, per phase
		self.ent = np.zeros((N_PHASES, 2), np.float64)     # per phase: [sum of pi entropy, sum of max(pi)]
		self.hand = np.zeros(N_PHASES, np.float64)         # sum of the hand size of the player to move
		self.round_hist = np.zeros(MAX_ROUNDS + 2, np.int64)   # positions per round -> game length
		self.played = np.zeros((N_PHASES, N_BLOCKS), np.int64) # the move actually PLAYED, per phase x action block
		self._armed = {}                                   # thread id -> a target was just recorded
		self.recv = np.zeros(N_TRADE_SETS, np.float64)     # pi mass per asked set (MAIN openings + counters)
		self.give = np.zeros(N_TRADE_SETS, np.float64)     # pi mass per offered set (GIVE ply)
		self.recv_legal = np.zeros(N_TRADE_SETS, np.float64)   # how often each set was LEGAL to ask for
		self.give_legal = np.zeros(N_TRADE_SETS, np.float64)   # ... and to offer: the mass above is only
		                                                       # readable against this availability baseline
		self.pair = np.zeros((N_TRADE_SETS, N_TRADE_SETS), np.float64)   # [asked, offered] mass on the GIVE ply
		self.answer = np.zeros(3, np.float64)              # trANSWER: [OK, NO, counter] mass
		self.answer_pair = np.zeros((2, N_TRADE_SETS, N_TRADE_SETS), np.float64)   # [OK/NO][asked][offered]
		self.accept = np.zeros(NUMBER_PLAYERS, np.float64) # trACCEPT: mass per relative player (0 = refuse all)
		import atexit
		atexit.register(self.flush)

	def _offer_of(self, board, rel_player):
		d = ROW_PLAYER + ROWS_PER_PLAYER * rel_player + 3
		recv = tuple(int(x) for x in board[d, PD_TRADE_RECV:PD_TRADE_RECV + N_RESOURCES])
		give = tuple(int(x) for x in board[d, PD_TRADE_GIVE:PD_TRADE_GIVE + N_RESOURCES])
		return self.set_index.get(recv, -1), self.set_index.get(give, -1)

	def record(self, board, pi, valids):
		pi = np.asarray(pi, dtype=np.float64)
		phase = int(board[ROW_GLOBAL, GA_PHASE])
		self.n_pos[phase] += 1
		self.legal[phase] += int(valids.sum())
		nz = pi[pi > 0]
		self.ent[phase, 0] += float(-(nz * np.log(nz)).sum())
		self.ent[phase, 1] += float(pi.max())
		self.hand[phase] += int(board[ROW_PLAYER, PA_TOTAL_RES])
		r = int(board[ROW_GLOBAL + 1, GB_ROUND_HI]) * 100 + int(board[ROW_GLOBAL + 1, GB_ROUND_LO])
		self.round_hist[min(max(r, 0), MAX_ROUNDS + 1)] += 1
		# getNextState() is called right after, by Coach/Arena only (MCTS goes
		# straight to the numba make_move), and on the very board this target was
		# built for: arming here is what tells the two apart, per thread.
		self._armed[threading.get_ident()] = True
		if phase == PHASE_MAIN:
			if valids[A_TRADE_RECV:A_TRADE_RECV + N_TRADE_SETS].any():
				self.main[0] += 1
				mass = pi[A_TRADE_RECV:A_TRADE_RECV + N_TRADE_SETS]
				self.main[1] += mass.sum()
				self.main[2] += A_TRADE_RECV <= int(pi.argmax()) < A_TRADE_RECV + N_TRADE_SETS
				self.main[3] += int(valids[A_TRADE_RECV:A_TRADE_RECV + N_TRADE_SETS].sum())
				self.main[4] += int(valids.sum())
				self.recv_legal += valids[A_TRADE_RECV:A_TRADE_RECV + N_TRADE_SETS]
				self.recv += mass
		elif phase == PHASE_TRADE_OFFER and valids[A_TRADE_GIVE:A_TRADE_GIVE + N_TRADE_SETS].any():
			mass = pi[A_TRADE_GIVE:A_TRADE_GIVE + N_TRADE_SETS]
			self.give += mass
			self.give_legal += valids[A_TRADE_GIVE:A_TRADE_GIVE + N_TRADE_SETS]
			asked, _ = self._offer_of(board, 0)             # the composer is the player to move
			if asked >= 0:
				self.pair[asked] += mass
		elif phase == PHASE_TRADE_ANSWER:
			self.answer[0] += pi[A_TRADE_OK]
			self.answer[1] += pi[A_TRADE_NO]
			counter = pi[A_TRADE_RECV:A_TRADE_RECV + N_TRADE_SETS]
			self.answer[2] += counter.sum()
			self.recv += counter
			asked, offered = self._offer_of(board, int(board[ROW_GLOBAL + 1, GB_TURN_PLAYER]))
			if asked >= 0 and offered >= 0:
				self.answer_pair[0, asked, offered] += pi[A_TRADE_OK]
				self.answer_pair[1, asked, offered] += pi[A_TRADE_NO]
		elif phase == PHASE_TRADE_ACCEPT:
			self.accept += pi[A_TRADE_ACCEPT:A_TRADE_ACCEPT + NUMBER_PLAYERS]
		self.n += 1
		if self.n % self.flush_every == 0:
			self.flush()

	def record_played(self, board, action):
		"""The move Coach actually sampled from the target recorded just above.
		Unarmed (arena, pit, PCR fast searches) it counts nothing, so `played`
		and `n_pos` always describe the same positions."""
		if not self._armed.pop(threading.get_ident(), False):
			return
		self.played[int(board[ROW_GLOBAL, GA_PHASE]), int(ACTION_BLOCK_OF[action])] += 1

	def flush(self):
		if self.n == 0:
			return
		np.savez(self.path, n_pos=self.n_pos, main=self.main, recv=self.recv, give=self.give, pair=self.pair,
		         answer=self.answer, answer_pair=self.answer_pair, accept=self.accept, legal=self.legal,
		         ent=self.ent, hand=self.hand, round_hist=self.round_hist, played=self.played,
		         recv_legal=self.recv_legal, give_legal=self.give_legal)


_trade_stats = _TradeStats(os.environ['CATAN_TRADE_STATS']) if os.environ.get('CATAN_TRADE_STATS') else None


class CatanGame(Game):
	def __init__(self):
		self.board = Board(NUMBER_PLAYERS)
		self.num_players = NUMBER_PLAYERS

	def getInitBoard(self):
		self.board.init_game()
		return self.board.get_state()

	def getBoardSize(self):
		return observation_size()

	def getActionSize(self):
		return action_size()

	def getNextState(self, board, player, action, random_seed=0):
		if _trade_stats is not None:
			_trade_stats.record_played(board, action)
		self.board.copy_state(board, True)
		next_player = self.board.make_move(action, player, random_seed)
		return (self.board.get_state(), next_player)

	def getValidMoves(self, board, player):
		self.board.copy_state(board, False)
		return self.board.valid_moves(player)

	def getGameEnded(self, board, next_player):
		self.board.copy_state(board, False)
		return self.board.check_end_game(next_player)

	def getScore(self, board, player):
		self.board.copy_state(board, False)
		return self.board.get_score(player)

	def getRound(self, board):
		self.board.copy_state(board, False)
		return self.board.get_round()

	def getCanonicalForm(self, board, player):
		if player == 0:
			return board

		self.board.copy_state(board, True)
		self.board.swap_players(player)
		return self.board.get_state()

	def getSymmetries(self, board, pi, valid_actions):
		if _trade_stats is not None:
			_trade_stats.record(board, pi, valid_actions)
		pi = np.array(pi, dtype=np.float32)
		# The net is equivariant by construction (CatanNNetTest), so the 12 isometries
		# of a position give the SAME loss and the SAME gradient: 11 of them are dead
		# weight. Keep one, and keep near-forced positions down-weighted by dropping
		# them at random instead of by generating fewer copies -- same sampling
		# weights as N_SYM_TRIVIAL/N_ISOMETRIES, 12x fewer samples.
		if int(valid_actions.sum()) <= SYM_TRIVIAL_MAX_LEGAL and np.random.random() >= N_SYM_TRIVIAL / N_ISOMETRIES:
			return []
		return [(np.array(board, copy=True), pi, valid_actions.copy())]

	def stringRepresentation(self, board):
		return board.tobytes()

	def getNumberOfPlayers(self):
		return NUMBER_PLAYERS

	def moveToString(self, move, current_player):
		return move_to_str(move, current_player)

	def printBoard(self, numpy_board):
		board = Board(self.getNumberOfPlayers())
		board.copy_state(numpy_board, False)
		print_board(board)

	# --- Hidden information (PIMC, see MCTS.py) ---------------------------
	# Both take/return canonical boards. `viewer` is the index, in the canonical
	# frame, of the player whose information set is wanted (0 = player to move).
	# Only opponents' resource and development cards are hidden; their hand SIZES
	# stay public.
	def getObservation(self, board, viewer):
		self.board.copy_state(board, True)
		self.board.get_observation(viewer)
		return self.board.get_state()

	def sampleWorld(self, observation, random_seed):
		self.board.copy_state(observation, True)
		self.board.sample_world(random_seed)
		return self.board.get_state()

	# DIAGNOSTIC ONLY (pit.py --peek, see MCTS.py): the opponents' RESOURCES stay
	# visible, only their dev cards are masked. Feeding this to sampleWorld gives
	# a world with the true resource hands. Not a legal information set.
	def getPeekObservation(self, board, viewer):
		self.board.copy_state(board, True)
		self.board.get_peek_observation(viewer)
		return self.board.get_state()
