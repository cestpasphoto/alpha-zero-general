import numpy as np
import random
import os
import sys
import termios
import tty
import select
from colorama import Style, Fore, Back

from .AbaloneLogicNumba import _decode_action, DIRECTIONS

def getch():
    """
    Reads a single keystroke from the terminal safely.
    Uses os.read to bypass Python's stdin buffering which causes deadlocks.
    """
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
        selected_marbles = []
        
        # --- INIT: Detect AI moves before the while loop ---
        new_arrivals = set()
        if self.last_board_seen is not None:
            for r in range(9):
                for q in range(9):
                    if (board[r, q, 0] == 1 and self.last_board_seen[r, q, 0] == 0) or \
                       (board[r, q, 1] == 1 and self.last_board_seen[r, q, 1] == 0):
                        new_arrivals.add((r, q))

        while True:
            # Dynamically compute valid destination cells
            possible_dests = set()
            selected_set = set(selected_marbles)
            
            if len(selected_set) > 0:
                for a in valid_moves:
                    r, q, size, axis, d = _decode_action(a)
                    group = set()
                    for i in range(size):
                        group.add((r + i * DIRECTIONS[axis, 0], q + i * DIRECTIONS[axis, 1]))
                    
                    if group == selected_set:
                        for gr, gq in group:
                            dest = (gr + DIRECTIONS[d, 0], gq + DIRECTIONS[d, 1])
                            if dest not in selected_set:
                                possible_dests.add(dest)

            # Pass new_arrivals to render
            self._render(board, cursor_r, cursor_q, selected_marbles, possible_dests, new_arrivals)
            
            k = getch()
            if k == '\x03': # Ctrl+C
                print("\nExiting...")
                sys.exit(0)
            
            # Navigation
            if k == '\x1b[A':   cursor_r = max(0, cursor_r - 1)
            elif k == '\x1b[B': cursor_r = min(8, cursor_r + 1)
            elif k == '\x1b[C': cursor_q = min(8, cursor_q + 1)
            elif k == '\x1b[D': cursor_q = max(0, cursor_q - 1)
            
            # Escape or Backspace to clear selection
            elif k == '\x1b' or k == '\x7f':
                selected_marbles.clear()
                
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
                
                # 2. Execution
                elif len(selected_marbles) > 0:
                    matched_action = None
                    for a in valid_moves:
                        r, q, size, axis, d = _decode_action(a)
                        group = set()
                        for i in range(size):
                            group.add((r + i * DIRECTIONS[axis, 0], q + i * DIRECTIONS[axis, 1]))
                        
                        if group == selected_set:
                            dest_group = set()
                            for gr, gq in group:
                                dest_group.add((gr + DIRECTIONS[d, 0], gq + DIRECTIONS[d, 1]))
                            if (cursor_r, cursor_q) in dest_group:
                                matched_action = a
                                break
                    
                    if matched_action is not None:
                        print("\033[H\033[J", end="") 
                        next_board, _ = self.game.getNextState(board, 0, matched_action)
                        self.last_board_seen = next_board.copy()
                        return matched_action

    def _render(self, board, cr, cq, selected_marbles, possible_dests, new_arrivals):
        print("\033[H\033[J", end="") 
        print("=" * 55)
        print(f" {Fore.GREEN}SPACE{Style.RESET_ALL}   : Select/Deselect your marbles (max 3)")
        print(f" {Fore.MAGENTA}SPACE{Style.RESET_ALL}   : Click on a MAGENTA cell to push")
        print(f" {Fore.CYAN}R{Style.RESET_ALL}       : Undo (Annuler les 2 derniers coups)")
        print(f" {Fore.YELLOW}ESC{Style.RESET_ALL}     : Clear selection")
        print("=" * 55)
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
        
        initial_score = self.game.getScore(board, 0)
        initial_opp_score = self.game.getScore(board, 1)

        for m in np.flatnonzero(valids):
            nextBoard, _ = self.game.getNextState(board, 0, m)
            my_score = self.game.getScore(nextBoard, 1) # Because swap_players was applied
            opp_score = self.game.getScore(nextBoard, 0)
            
            score_diff = my_score - opp_score
            candidates.append((score_diff, m))
            
        max_score = max(candidates, key=lambda x: x[0])[0]
        actions_leading_to_max = [m for (s, m) in candidates if s == max_score]
        
        return random.choice(actions_leading_to_max)