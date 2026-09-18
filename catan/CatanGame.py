import sys
sys.path.append('..')
from Game import Game
from .CatanConstants import N_PLAYERS as NUMBER_PLAYERS
from .CatanLogicNumba import Board, observation_size, action_size
from .CatanDisplay import move_to_str, print_board
import numpy as np

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
		self.board.copy_state(board, True)
		return self.board.get_symmetries(np.array(pi, dtype=np.float32), valid_actions)

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
