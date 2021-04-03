import sys
sys.path.append('..')
from Game import Game
from .SplendorLogic import observation_size, action_size, move_to_short_str, print_board
from .SplendorLogicNumba import Board
import numpy as np
from numba import jit, njit

# (Game convention) -1 = 1 (SplendorLogic convention)
# (Game convention)  1 = 0 (SplendorLogic convention)

@njit(fastmath=True) # No cache, because relies jitclass which isn't compatible with cache
def getGameEnded(splendorgameboard, board, player):
	splendorgameboard.copy_state(board, False)
	ended, winners = splendorgameboard.check_end_game()
	np_winners = np.array(winners, dtype=np.bool_)
	if not ended:			# not finished
		return 0.

	if np_winners.sum() != 1:   # finished but no winners or several
		return 0.01
	return 1. if np_winners[0 if player==1 else 1] else -1.

@njit(fastmath=True)
def getNextState(splendorgameboard, board, player, action, deterministic=False):
	splendorgameboard.copy_state(board, True)
	splendorgameboard.make_move(action, 0 if player==1 else 1, deterministic)
	return (splendorgameboard.get_state(), -player)

@njit(fastmath=True)
def getValidMoves(splendorgameboard, board, player):
	splendorgameboard.copy_state(board, False)
	return splendorgameboard.valid_moves(0 if player==1 else 1)

@njit(fastmath=True)
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


def generate_board(n_moves=10):
	game = SplendorGame(2)
	board = game.getInitBoard()
	curPlayer = 1
	for _ in range(n_moves):
		canonicalBoard = game.getCanonicalForm(board, curPlayer)
		valids = game.getValidMoves(canonicalBoard, 1)
		uniform_pi = [ int(v) / sum(valids) for v in valids ]
		action = np.random.choice(range(len(uniform_pi)), p=uniform_pi)
		board, curPlayer = game.getNextState(board, curPlayer, action)

	return game, board

def test_game(generate_test=False):
	if generate_test:
		import copy
		import random
		from .SplendorLogic import print_board, move_to_short_str, row_to_str, Board
		print('Searching convenient starting board')
		while True:
			try:
				game, board = generate_board()
				random_state = random.getstate()
				game_copy = copy.deepcopy(game)
				board_copy = np.copy(board)
				list_actions = [2, 12, 31, 27, 65, 80, 26, 79, 59, 80]

				curPlayer = 1
				trainExamples = []
				output_data, input_data = [], []
				for action in list_actions:
					canonical_board = game.getCanonicalForm(board, curPlayer)
					canonical_board_copy = np.copy(canonical_board)
					valids = game.getValidMoves(canonical_board, 1)
					if valids[action] == 0:
						raise Warning('invalid action')
					pi = [ i/100 for i,v in enumerate(valids) ] # Fake policy
					pi_copy = np.copy(pi)
					sym = game.getSymmetries(canonical_board, pi, valids)
					for b, p, v in sym:
						trainExamples.append([b, curPlayer, p, v])
					board, curPlayer = game.getNextState(board, curPlayer, action)

					# Check nothing has changed
					if len(output_data) > 0:
						for old_ex, ex in zip(output_data[-1]['trainExamples'], trainExamples[:-1]):
							b, curPlayer, p, v = ex
							old_b, old_curPlayer, old_p, old_v = old_ex
							if not np.array_equal(b, old_b) or curPlayer != old_curPlayer or not np.array_equal(p, old_p) or not np.array_equal(v, old_v):
								breakpoint()
					if not np.array_equal(canonical_board, canonical_board_copy) or not np.array_equal(pi, pi_copy):
						breakpoint()
					
					input_data.append({
						'action': action,
						'policy': pi_copy,
					})
					output_data.append({
						'canonical_board': canonical_board_copy,
						'trainExamples': copy.deepcopy(trainExamples),
					})

			except Warning:
				continue
			else:
				print('Success')
				import pickle
				with open('test_data.pickle', 'wb') as f:
					pickle.dump({
						'random_state': random_state,
						'initial_game': game_copy,
						'initial_board': board_copy,
						'input_data': input_data,
						'output_data': output_data,
					}, f)

				# Review
				for inputs, outputs in zip(input_data, output_data):
					gameboard = Board(2, outputs['canonical_board'])
					print_board(gameboard)
					_ = input('You confirm this is ok ? ')
					print('---> ', move_to_short_str(inputs['action']), '(', inputs['action'], ')')

				for example_index in [-2, -1]:
					last_examples = output_data[example_index]['trainExamples']
					start_index = len(output_data[example_index-1]['trainExamples'])
					last_examples = last_examples[start_index:]
					for i,example in enumerate(last_examples):
						if i == 0:
							continue
						print()
						print('REFERENCE BOARD IS:')
						ref_gameboard = Board(2, last_examples[0][0])
						print_board(ref_gameboard)

						print('SYMMETRY IS:')
						gameboard = Board(2, example[0])
						print_board(gameboard)
						print('curPlayer =', example[1])

						for j in range(len(example[2])):
							if abs( example[2][j] - last_examples[0][2][j] ) > 0.001 or example[3][j] != last_examples[0][3][j]:
								print(f'Move {j}: {round(last_examples[0][2][j]*100):<03} {last_examples[0][3][j]}\t{round(example[2][j]*100):<03} {example[3][j]}\t{move_to_short_str(j)}')
						_ = input('You confirm this is ok ? ')

				break

	else:
		from .SplendorLogic import print_board, move_to_short_str, row_to_str
		import pickle
		import random
		import time
		with open('test_data.pickle', 'rb') as f:
			test_data = pickle.load(f)
		game, board = test_data['initial_game'], test_data['initial_board']
		random.setstate(test_data['random_state'])

		start_time = time.time()

		curPlayer = 1
		trainExamples = []
		for inputs, outputs in zip(test_data['input_data'], test_data['output_data']):
			canonical_board = game.getCanonicalForm(board, curPlayer)
			canonical_board_copy = np.copy(canonical_board)
			valids = game.getValidMoves(canonical_board, 1)
			action = inputs['action']
			if valids[action] == 0:
				raise Exception('invalid action')
			pi = inputs['policy']
			pi_copy = np.copy(pi)
			sym = game.getSymmetries(canonical_board, pi, valids)
			for b, p, v in sym:
				trainExamples.append([b, curPlayer, p, v])
			board, curPlayer = game.getNextState(board, curPlayer, action)
			# print_board(game.board)

			# Check nothing has changed
			if not np.array_equal(canonical_board, canonical_board_copy) or not np.array_equal(pi, pi_copy):
				print('!! VALUES HAVE CHANGED DURING EXECUTION')
				breakpoint()
			if not np.array_equal(canonical_board, outputs['canonical_board']):
				for x in range(canonical_board.shape[0]):
					if np.any(canonical_board[x] != outputs['canonical_board'][x]):
						print(f'canonical_boards diff on row {row_to_str(x)} ({x}):')
						print(' ', canonical_board[x])
						print(' ', outputs['canonical_board'][x])
			# CHECKS
			if len(trainExamples) != len(outputs['trainExamples']):
				print(f'Different examples lengths: {len(trainExamples)} {len(outputs["trainExamples"])}')
			for te, exp_te in zip(trainExamples, outputs['trainExamples']):
				b, cp, p, v = te
				exp_b, exp_cp, exp_p, exp_v = exp_te
				if not np.array_equal(b, exp_b):
					print('Differences in board in trainExamples')
					for x in range(b.shape[0]):
						if np.any(b[x] != exp_b[x]):
							print(f'trainExamples board diff on row {row_to_str(x)} ({x}):')
							print(' ', b[x])
							print(' ', exp_b[x])
				elif cp != exp_cp:
					print('Differences in curPlayer in trainExamples')
				elif not np.array_equal(p, exp_p):
					print('Differences in policy in trainExamples')
				elif not np.array_equal(v, exp_v):
					print('Differences in valids in trainExamples')
				elif b.dtype != np.int8 or p.dtype != np.float32 or v.dtype != np.bool_:
					print('Incorrect types')
				else:
					continue
				breakpoint()
				print('At least an issue happened')

		duration = time.time() - start_time
		print(round(duration*1000000))


