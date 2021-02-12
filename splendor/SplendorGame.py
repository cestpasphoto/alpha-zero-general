import sys
sys.path.append('..')
from Game import Game
from .SplendorLogic import Board, observation_size, action_size
import numpy as np

# (Game convention) -1 = 1 (SplendorLogic convention)
# (Game convention)  1 = 0 (SplendorLogic convention)

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

	def getNextState(self, board, player, action):
		self.board.copy_state(board)
		self.board.make_move(action, 0 if player==1 else 1)
		return (self.board.get_state(), -player)


	def getValidMoves(self, board, player):
		self.board.copy_state(board)
		return self.board.valid_moves(0 if player==1 else 1)

	def getGameEnded(self, board, player):
		self.board.copy_state(board)
		ended, winners = self.board.check_end_game()
		if not ended:			# not finished
			return 0

		if sum(winners) != 1:   # finished but no winners or several
			return 0.01
		return 1 if winners[0 if player==1 else 1] else -1

	def getScore(self, board, player):
		self.board.copy_state(board)
		return self.board.get_score(0 if player==1 else 1)


	def getCanonicalForm(self, board, player):
		if player == 1:
			return board

		self.board.copy_state(board)
		self.board.swap_players()
		return self.board.get_state()
		

	def getSymmetries(self, board, pi, valid_actions):
		self.board.copy_state(board)
		return self.board.get_symmetries(pi, valid_actions)

	def stringRepresentation(self, board):
		return board.tobytes()
