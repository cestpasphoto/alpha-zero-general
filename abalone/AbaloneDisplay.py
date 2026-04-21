import numpy as np
from colorama import Style, Fore, Back
from .AbaloneLogicNumba import _decode_action, DIRECTIONS

# Human-readable names for UI mapping
DIR_NAMES = ["East", "South-East", "South-West", "West", "North-West", "North-East"]
AXIS_NAMES = ["E-W", "NW-SE", "NE-SW"]

def move_to_str(move, player):
	"""
	Translates a 1D action ID into a human-readable format.
	"""
	r, q, size, axis, d = _decode_action(move)
	dir_name = DIR_NAMES[d]
	
	if size == 1:
		return f"Move marble at (r:{r}, q:{q}) towards {dir_name}"
	else:
		axis_name = AXIS_NAMES[axis]
		return f"Move {size} marbles starting at (r:{r}, q:{q}) along {axis_name} towards {dir_name}"

def _print_round_and_scores(board):
	print("=" * 45)
	print(f" Round: {board.get_round()} ")
	print(f" {Fore.RED}Player 0 (Black){Style.RESET_ALL} score: {board.get_score(0)}/6")
	print(f" {Fore.WHITE}Player 1 (White){Style.RESET_ALL} score: {board.get_score(1)}/6")
	print("=" * 45)

def _print_hex_grid(board):
	"""
	Prints the axial grid as a proper hexagon in the terminal.
	Because r=0 has 5 cells, r=4 has 9 cells, etc., we pad the left 
	side with spaces proportional to the distance from the center row (r=4).
	"""
	for r in range(9):
		# Calculate padding to form the hexagonal shape
		spaces = abs(r - 4)
		print(" " * spaces, end="")
		
		for q in range(9):
			if board.board_mask[r, q] == 1:
				if board.my_marbles[r, q] == 1:
					# Player 0 (Usually Black, printing in RED for dark terminal visibility)
					print(f"{Fore.RED} ⬤ {Style.RESET_ALL}", end="")
				elif board.opp_marbles[r, q] == 1:
					# Player 1 (White)
					print(f"{Fore.WHITE} ⬤ {Style.RESET_ALL}", end="")
				else:
					# Empty playable hole
					print(f"{Fore.LIGHTBLACK_EX} + {Style.RESET_ALL}", end="")
		print() # New line for the next row
	print()

def print_board(board):
	print()
	_print_round_and_scores(board)
	_print_hex_grid(board)