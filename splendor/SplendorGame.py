import sys
sys.path.append('..')
from Game import Game
from .SplendorLogic import observation_size, action_size, move_to_short_str, print_board
from .SplendorLogicNumba import Board
import numpy as np
from numba import jit, njit

# (Game convention) -1 = 1 (SplendorLogic convention)
# (Game convention)  1 = 0 (SplendorLogic convention)

@njit(fastmath=True, nogil=True) # No cache, because relies jitclass which isn't compatible with cache
def getGameEnded(splendorgameboard, board, player):
	splendorgameboard.copy_state(board, False)
	ended, winners = splendorgameboard.check_end_game()
	np_winners = np.array(winners, dtype=np.bool_)
	if not ended:			# not finished
		return 0.

	if np_winners.sum() != 1:   # finished but no winners or several
		return 0.01
	return 1. if np_winners[0 if player==1 else 1] else -1.

@njit(fastmath=True, nogil=True)
def getNextState(splendorgameboard, board, player, action, deterministic=False):
	splendorgameboard.copy_state(board, True)
	splendorgameboard.make_move(action, 0 if player==1 else 1, deterministic)
	return (splendorgameboard.get_state(), -player)

@njit(fastmath=True, nogil=True)
def getValidMoves(splendorgameboard, board, player):
	splendorgameboard.copy_state(board, False)
	return splendorgameboard.valid_moves(0 if player==1 else 1)

@njit(fastmath=True, nogil=True)
def getCanonicalForm(splendorgameboard, board, player):
	if player == 1:
		return board

	splendorgameboard.copy_state(board, True)
	splendorgameboard.swap_players()
	return splendorgameboard.get_state()

class SplendorGame(Game):
	def __init__(self, num_players):
		self.num_players = num_players
		self.board = Board(num_players)

	def getInitBoard(self):
		self.board.init_game()
		return self.board.get_state()

	def getBoardSize(self):
		return observation_size(self.num_players)

	def getActionSize(self):
		return action_size()

	def getMaxScoreDiff(self):
		return 15

	def getNextState(self, board, player, action, deterministic=False):
		self.board.copy_state(board, True)
		self.board.make_move(action, 0 if player==1 else 1, deterministic)
		return (self.board.get_state(), -player)


	def getValidMoves(self, board, player):
		self.board.copy_state(board, False)
		return self.board.valid_moves(0 if player==1 else 1)

	def getGameEnded(self, board, player):
		self.board.copy_state(board, False)
		ended, winners = self.board.check_end_game()
		if not ended:			# not finished
			return 0

		if sum(winners) != 1:   # finished but no winners or several
			return 0.01
		return 1 if winners[0 if player==1 else 1] else -1

	def getScore(self, board, player):
		self.board.copy_state(board, False)
		return self.board.get_score(0 if player==1 else 1)

	def getRound(self, board):
		self.board.copy_state(board, False)
		return self.board.get_round()

	def getCanonicalForm(self, board, player):
		if player == 1:
			return board

		self.board.copy_state(board, True)
		self.board.swap_players()
		return self.board.get_state()

	def getSymmetries(self, board, pi, valid_actions):
		self.board.copy_state(board, True)
		return self.board.get_symmetries(np.array(pi, dtype=np.float32), valid_actions)

	def stringRepresentation(self, board):
		return board.tobytes()
