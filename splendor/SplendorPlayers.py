import numpy as np
import random

class RandomPlayer():
	def __init__(self, game):
		self.game = game

	def play(self, board):
		valids = self.game.getValidMoves(board, 1)
		return random.choices(range(self.game.getActionSize()), weights=valids, k=1)[0]


# class HumanOthelloPlayer():
# 	def __init__(self, game):
# 		self.game = game

# 	def play(self, board):
# 		# display(board)
# 		valid = self.game.getValidMoves(board, 1)
# 		for i in range(len(valid)):
# 			if valid[i]:
# 				print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
# 		while True:
# 			input_move = input()
# 			input_a = input_move.split(" ")
# 			if len(input_a) == 2:
# 				try:
# 					x,y = [int(i) for i in input_a]
# 					if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
# 							((x == self.game.n) and (y == 0)):
# 						a = self.game.n * x + y if x != -1 else self.game.n ** 2
# 						if valid[a]:
# 							break
# 				except ValueError:
# 					# Input needs to be an integer
# 					'Invalid integer'
# 			print('Invalid move')
# 		return a


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