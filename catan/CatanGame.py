import sys
sys.path.append('..')
from Game import Game
from .CatanConstants import N_PLAYERS as NUMBER_PLAYERS
from .CatanConstants import N_ISOMETRIES, SYM_TRIVIAL_MAX_LEGAL, N_SYM_TRIVIAL
from .CatanLogicNumba import Board, observation_size, action_size
from .CatanDisplay import move_to_str, print_board
import numpy as np

# MCTS.search() is recursive (one Python frame per ply) and a Catan game runs up
# to 4*MAX_ROUNDS=1600 decisions, well past Python's default limit of 1000.
# This raises Python's counter only: if a hard crash (segfault) ever appears,
# the OS stack is too small, e.g. `ulimit -s 65532` before launching python.
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
		# The net is equivariant by construction (CatanNNetTest), so the 12
		# isometries of a position give the same gradient: keep one. Near-forced
		# positions are down-weighted by dropping them at random, with the same
		# sampling weight N_SYM_TRIVIAL/N_ISOMETRIES.
		pi = np.array(pi, dtype=np.float32)
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

	# --- Hidden information (see MCTS.py) -----------------------------------
	# Both take/return canonical boards. `viewer` is the index, in the canonical
	# frame, of the player whose information set is wanted (0 = player to move).
	# Hidden: opponents' resource and development cards, and the composition of
	# the development deck. Hand sizes stay public.
	def getObservation(self, board, viewer):
		self.board.copy_state(board, True)
		self.board.get_observation(viewer)
		return self.board.get_state()

	def sampleWorld(self, observation, random_seed):
		self.board.copy_state(observation, True)
		self.board.sample_world(random_seed)
		return self.board.get_state()
