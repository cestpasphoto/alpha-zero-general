import numpy as np
import random

from .SplendorLogic import print_board, move_to_short_str

class RandomPlayer():
	def __init__(self, game):
		self.game = game

	def play(self, board):
		valids = self.game.getValidMoves(board, 1)
		return random.choices(range(self.game.getActionSize()), weights=valids, k=1)[0]


class HumanPlayer():
	def __init__(self, game):
		self.game = game

	def play(self, board):
		# print_board(self.game.board)
		valid = self.game.getValidMoves(board, 1)
		for i, v in enumerate(valid):
			if i in [12,12+15,12+15+3+30]:
				print()
			if v:
				print(f'{i} = {move_to_short_str(i)}', end='   ')
		print()
		while True:
			input_move = input()
			try:
				a = int(input_move)
				if not valid[a]:
					raise Exception('')
				break
			except:
				print('Invalid move:', input_move)
		return a


class GreedyPlayer():
	def __init__(self, game):
		self.game = game

	def play(self, board):
		valids = self.game.getValidMoves(board, 1)
		candidates = []
		initial_score = self.game.getScore(board, 1)
		for m in [m_ for m_, v in enumerate(valids) if v>0]:
			nextBoard, _ = self.game.getNextState(board, 1, m)
			score = self.game.getScore(nextBoard, 1)
			candidates += [(score, m)]
		max_score = max(candidates, key=lambda x: x[0])[0]
		if max_score == initial_score:
			actions_leading_to_max = [m for (m, v) in enumerate(valids) if v>0 and 0      <=m<12        ]
			if len(actions_leading_to_max) == 0:
				actions_leading_to_max = [m for (m, v) in enumerate(valids) if v>0 and 12+15+3<=m<12+15+3+30]
				if len(actions_leading_to_max) == 0:
					actions_leading_to_max = [m for (m, v) in enumerate(valids) if v>0]
		else:
			actions_leading_to_max = [m for (s,m) in candidates if s==max_score]
		move = random.choice(actions_leading_to_max)
		return move