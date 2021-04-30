import sys
sys.path.append('..')
from Game import Game
from .SplendorLogic import observation_size, action_size, print_board
from .SplendorLogicNumba import Board
import numpy as np
from numba import jit, njit

# (Game convention) -1 = 1 (SplendorLogic convention)
# (Game convention)  1 = 0 (SplendorLogic convention)

@njit(fastmath=True, nogil=True) # No cache, because relies jitclass which isn't compatible with cache
def getGameEnded(splendorgameboard, board):
	splendorgameboard.copy_state(board, False)
	return splendorgameboard.check_end_game()

@njit(fastmath=True, nogil=True)
def getNextState(splendorgameboard, board, player, action, deterministic=False):
	splendorgameboard.copy_state(board, True)
	splendorgameboard.make_move(action, player, deterministic)
	return (splendorgameboard.get_state(), (player+1)%splendorgameboard.num_players)

@njit(fastmath=True, nogil=True)
def getValidMoves(splendorgameboard, board, player):
	splendorgameboard.copy_state(board, False)
	return splendorgameboard.valid_moves(player)

@njit(fastmath=True, nogil=True)
def getCanonicalForm(splendorgameboard, board, player):
	if player == 0:
		return board

	splendorgameboard.copy_state(board, True)
	splendorgameboard.swap_players(player)
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
		self.board.make_move(action, player, deterministic)
		return (self.board.get_state(), (player+1)%self.num_players)


	def getValidMoves(self, board, player):
		self.board.copy_state(board, False)
		return self.board.valid_moves(player)

	def getGameEnded(self, board):
		self.board.copy_state(board, False)
		return self.board.check_end_game()

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
