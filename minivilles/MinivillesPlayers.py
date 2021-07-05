import numpy as np
import random

from .MinivillesDisplay import print_board, move_to_str

class RandomPlayer():
	def __init__(self, game):
		self.game = game

	def play(self, board, player=0):
		valids = self.game.getValidMoves(board, player)
		action = random.choices(range(self.game.getActionSize()), weights=valids.astype(np.int), k=1)[0]
		return action


class HumanPlayer():
	def __init__(self, game):
		self.game = game

	def show_all_moves(self, valid):
		for i, v in enumerate(valid):
			if v:
				print(f'{i} = {move_to_str(i)}', end='   ')
		print()

	def play(self, board):
		# print_board(self.game.board)
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
				except:
					print('Invalid move:', input_move)
		return a


class GreedyPlayer():
	pass