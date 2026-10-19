import numpy as np
import random
import os
import sys
from colorama import Style, Fore, Back

from .AbaloneLogicNumba import _decode_action, DIRECTIONS

def getch():
	"""
	Reads a single keystroke from the terminal safely.
	Uses os.read to bypass Python's stdin buffering which causes deadlocks.
	"""
	# Unix-only modules, imported lazily so that the module still imports elsewhere
	import termios, tty, select
	fd = sys.stdin.fileno()
	old_settings = termios.tcgetattr(fd)
	try:
		tty.setraw(fd)
		# Read raw bytes directly from the OS file descriptor
		ch_bytes = os.read(fd, 1)
		ch = ch_bytes.decode('utf-8', 'ignore')

		# If escape character is detected, check for arrow keys sequences
		if ch == '\x1b':
			if select.select([fd], [], [], 0.05)[0]:
				ch += os.read(fd, 1).decode('utf-8', 'ignore')
				if select.select([fd], [], [], 0.05)[0]:
					ch += os.read(fd, 1).decode('utf-8', 'ignore')
	finally:
		termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

	return ch

def _row_bounds(r):
	# Playable cells of row r satisfy 4 <= r + q <= 12
	return max(0, 4 - r), min(8, 12 - r)

def _visual_col(r, q):
	# Rows are indented by |r - 4| spaces and cells are 2 chars wide
	return 2 * q + r

class RandomPlayer():
	def __init__(self, game):
		self.game = game

	def play(self, board, nb_moves):
		valids = self.game.getValidMoves(board, player=0)
		action = random.choices(range(self.game.getActionSize()), weights=valids.astype(np.int8), k=1)[0]
		return action

class HumanPlayer():
	def __init__(self, game):
		self.game = game
		self.last_board_seen = None # Memory for AI moves

	def play(self, board, nb_moves):
		valid = self.game.getValidMoves(board, 0)
		valid_moves = np.flatnonzero(valid)

		cursor_r, cursor_q = 8, 0
		target_col = _visual_col(cursor_r, cursor_q) # Remembered column for vertical moves
		selected_marbles = []
		alt_idx = 0 # Which move to play when several share the cursor cell

		# --- INIT: Detect AI moves before the while loop ---
		new_arrivals = set()
		if self.last_board_seen is not None:
			for r in range(9):
				for q in range(9):
					if (board[r, q, 0] == 1 and self.last_board_seen[r, q, 0] == 0) or \
					   (board[r, q, 1] == 1 and self.last_board_seen[r, q, 1] == 0):
						new_arrivals.add((r, q))

		while True:
			# Map each reachable destination cell to the list of (action, destination cells)
			selected_set = set(selected_marbles)
			move_options = {}
			if len(selected_set) > 0:
				for a in valid_moves:
					r, q, size, axis, d = _decode_action(a)
					group = set()
					for i in range(size):
						group.add((r + i * DIRECTIONS[axis, 0], q + i * DIRECTIONS[axis, 1]))

					if group == selected_set:
						dest_group = set((gr + DIRECTIONS[d, 0], gq + DIRECTIONS[d, 1]) for gr, gq in group)
						for dest in dest_group - selected_set:
							move_options.setdefault(dest, []).append((a, dest_group))
			possible_dests = set(move_options.keys())

			# Broadside moves can share a destination cell (e.g. NE and NW for an E-W pair):
			# preview the chosen one, TAB cycles between them
			options_here = move_options.get((cursor_r, cursor_q), [])
			chosen = options_here[alt_idx % len(options_here)] if options_here else None
			preview = chosen[1] if chosen is not None else set()

			self._render(board, cursor_r, cursor_q, selected_marbles, possible_dests, new_arrivals,
			             preview, alt_idx % max(1, len(options_here)), len(options_here))

			k = getch()
			if k == '\x03': # Ctrl+C
				print("\nExiting...")
				sys.exit(0)

			# Navigation, cursor always stays on a playable cell
			if k in ('\x1b[C', '\x1b[D'):
				lo, hi = _row_bounds(cursor_r)
				cursor_q = min(hi, cursor_q + 1) if k == '\x1b[C' else max(lo, cursor_q - 1)
				target_col = _visual_col(cursor_r, cursor_q)
				alt_idx = 0
			elif k in ('\x1b[A', '\x1b[B'):
				nr = cursor_r - 1 if k == '\x1b[A' else cursor_r + 1
				if 0 <= nr <= 8:
					lo, hi = _row_bounds(nr)
					# Closest cell to the remembered column, ties broken to the right
					cursor_q = min(range(lo, hi + 1), key=lambda qq: (abs(_visual_col(nr, qq) - target_col), -qq))
					cursor_r = nr
				alt_idx = 0

			# Cycle between moves sharing the cursor cell
			elif k == '\t':
				alt_idx += 1

			# Escape or Backspace to clear selection
			elif k == '\x1b' or k == '\x7f':
				selected_marbles.clear()
				alt_idx = 0

			# Undo (R)
			elif k == 'r' or k == 'R':
				print("\033[H\033[J", end="")
				self.last_board_seen = None # Clear memory so time travel doesn't create ghost artifacts
				return -1

			# Action (Space or Enter)
			elif k in [' ', '\r', '\n']:
				# 1. Selection
				if board[cursor_r, cursor_q, 0] == 1:
					if (cursor_r, cursor_q) in selected_marbles:
						selected_marbles.remove((cursor_r, cursor_q))
					elif len(selected_marbles) < 3:
						selected_marbles.append((cursor_r, cursor_q))
					alt_idx = 0

				# 2. Execution
				elif chosen is not None:
					matched_action = chosen[0]
					print("\033[H\033[J", end="")
					next_board, _ = self.game.getNextState(board, 0, matched_action)
					self.last_board_seen = next_board.copy()
					return matched_action

	def _render(self, board, cr, cq, selected_marbles, possible_dests, new_arrivals, preview, alt_pos, n_alts):
		print("\033[H\033[J", end="")
		print("=" * 55)
		print(f" {Fore.GREEN}SPACE{Style.RESET_ALL}   : Select/Deselect your marbles (max 3)")
		print(f" {Fore.MAGENTA}SPACE{Style.RESET_ALL}   : Click on a MAGENTA cell to push")
		print(f" {Fore.CYAN}TAB{Style.RESET_ALL}     : Switch between moves sharing this cell")
		print(f" {Fore.CYAN}R{Style.RESET_ALL}       : Undo last 2 moves")
		print(f" {Fore.YELLOW}ESC{Style.RESET_ALL}     : Clear selection")
		print("=" * 55)
		if n_alts > 1:
			print(f" Move {alt_pos + 1}/{n_alts} for this cell (CYAN = resulting position), TAB for next")
		else:
			print()

		for r in range(9):
			spaces = abs(r - 4)
			print(" " * spaces, end="")
			for q in range(9):
				if not (4 <= r + q <= 12):
					continue

				content_char = "+"
				color = Fore.LIGHTBLACK_EX

				# AI move feedback integration
				if board[r, q, 0] == 1:
					content_char = "⬤"
					color = Fore.RED
				elif board[r, q, 1] == 1:
					content_char = "⬤"
					color = Fore.WHITE

				bg_color = ""
				if r == cr and q == cq:
					bg_color = Back.YELLOW
				elif (r, q) in selected_marbles:
					bg_color = Back.GREEN
				elif (r, q) in preview:
					bg_color = Back.CYAN
				elif (r, q) in possible_dests:
					bg_color = Back.MAGENTA
				elif (r, q) in new_arrivals:
					bg_color = Back.LIGHTBLUE_EX

				print(f"{bg_color}{color}{content_char} {Style.RESET_ALL}", end="")
			print()
		print("\n")

class GreedyPlayer():
	def __init__(self, game):
		self.game = game

	def play(self, board, nb_moves):
		# Abalone is a zero-sum game, a basic greedy strategy evaluates
		# the board state primarily by checking the score differential.
		valids = self.game.getValidMoves(board, 0)
		candidates = []

		for m in np.flatnonzero(valids):
			nextBoard, _ = self.game.getNextState(board, 0, m)
			# getNextState does NOT swap players: layer/score 0 is still the mover
			my_score = self.game.getScore(nextBoard, 0)
			opp_score = self.game.getScore(nextBoard, 1)

			score_diff = my_score - opp_score
			candidates.append((score_diff, m))

		max_score = max(candidates, key=lambda x: x[0])[0]
		actions_leading_to_max = [m for (s, m) in candidates if s == max_score]

		return random.choice(actions_leading_to_max)
