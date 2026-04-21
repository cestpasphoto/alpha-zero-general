import numpy as np
import random
import sys
import termios
import tty
from colorama import Style, Fore, Back

from .AbaloneLogicNumba import _decode_action, DIRECTIONS

def getch():
    """
    Reads a single keystroke from the terminal without requiring the Enter key.
    Captures arrow keys as 3-character escape sequences (\x1b[A, etc.).
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        if ch == '\x1b': # Arrow keys start with an escape character
            ch += sys.stdin.read(2)
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

    def play(self, board, nb_moves):
        valid = self.game.getValidMoves(board, 0)
        valid_moves = np.flatnonzero(valid)
        
        cursor_r, cursor_q = 8, 0 # Start near Player 0's default layout
        phase = 0 # 0: Pick anchor, 1: Pick action direction
        available_actions = []
        action_idx = 0

        while True:
            self._render(board, cursor_r, cursor_q, phase, available_actions, action_idx)
            
            k = getch()
            if k == '\x03': # Ctrl+C
                print("\nExiting...")
                sys.exit(0)
            
            if phase == 0:
                if k == '\x1b[A':   cursor_r = max(0, cursor_r - 1) # Up
                elif k == '\x1b[B': cursor_r = min(8, cursor_r + 1) # Down
                elif k == '\x1b[C': cursor_q = min(8, cursor_q + 1) # Right
                elif k == '\x1b[D': cursor_q = max(0, cursor_q - 1) # Left
                elif k == ' ' or k == '\r':
                    # Filter all valid moves starting from the selected anchor
                    available_actions = [m for m in valid_moves if _decode_action(m)[0] == cursor_r and _decode_action(m)[1] == cursor_q]
                    if available_actions:
                        phase = 1
                        action_idx = 0
            
            elif phase == 1:
                if k == '\x1b[C' or k == '\x1b[B':   # Right / Down cycle forward
                    action_idx = (action_idx + 1) % len(available_actions)
                elif k == '\x1b[D' or k == '\x1b[A': # Left / Up cycle backward
                    action_idx = (action_idx - 1) % len(available_actions)
                elif k == '\x7f' or k == '\x1b':     # Backspace or Esc to cancel
                    phase = 0
                elif k == ' ' or k == '\r':          # Space or Enter to confirm
                    print("\033[H\033[J", end="")    # Final clear
                    return available_actions[action_idx]

    def _render(self, board, cr, cq, phase, available_actions, action_idx):
        # Clear terminal completely and put cursor at top-left
        print("\033[H\033[J", end="") 
        
        print("=" * 45)
        if phase == 0:
            print(f"{Fore.YELLOW} Phase 1: Use ARROWS to move, SPACE to select.{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN} Phase 2: ARROWS to cycle preview, SPACE to confirm, ESC to cancel.{Style.RESET_ALL}")
        print("=" * 45)
        print()

        preview_move = available_actions[action_idx] if phase == 1 else None
        group_cells = []
        dest_cells = []
        
        if preview_move is not None:
            r, q, size, axis, d = _decode_action(preview_move)
            for i in range(size):
                gr = r + i * DIRECTIONS[axis, 0]
                gq = q + i * DIRECTIONS[axis, 1]
                group_cells.append((gr, gq))
                dest_cells.append((gr + DIRECTIONS[d, 0], gq + DIRECTIONS[d, 1]))

        # Render the hex grid with interactive overlays
        for r in range(9):
            spaces = abs(r - 4)
            print(" " * spaces, end="")
            for q in range(9):
                if not (4 <= r + q <= 12):
                    # Out of bounds
                    print("   ", end="")
                    continue
                
                # Base content
                content = f"{Fore.LIGHTBLACK_EX} + {Style.RESET_ALL}"
                if board[r, q, 0] == 1:
                    content = f"{Fore.RED} ⬤ {Style.RESET_ALL}"
                elif board[r, q, 1] == 1:
                    content = f"{Fore.WHITE} ⬤ {Style.RESET_ALL}"
                
                # Overlays
                if phase == 0 and r == cr and q == cq:
                    content = f"{Back.YELLOW}{content}{Style.RESET_ALL}"
                elif phase == 1:
                    if (r, q) in group_cells:
                        content = f"{Back.GREEN}{content}{Style.RESET_ALL}"
                    elif (r, q) in dest_cells:
                        content = f"{Back.MAGENTA}{content}{Style.RESET_ALL}"

                print(content, end="")
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