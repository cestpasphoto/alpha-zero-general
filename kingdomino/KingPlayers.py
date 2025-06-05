import numpy as np
import random

from .KingLogic import print_board, move_to_str

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, _nb_moves):
        valids = self.game.getValidMoves(board, player=0)
        action = random.choices(range(self.game.getActionSize()), weights=valids.astype(np.int8), k=1)[0]
        return action


class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, nb_moves):
        valids = self.game.getValidMoves(board, 0)
        _ = input("Continue? ")
        action = random.choices(range(self.game.getActionSize()), weights=valids.astype(np.int8), k=1)[0]
        return action


class GreedyPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, _nb_moves):
        valids = self.game.getValidMoves(board, 0)
        best_move = None
        best_to_line = 0
        best_to_floor = 0
        for move in range(len(valids)):
            if valids[move]:
                factory = board[3 + (move // 30)]
                colour = (move % 30) // 6
                line = move % 6
                num_tiles = factory[colour]
                if line == 5:
                    to_line = 0
                    to_floor = num_tiles
                else:
                    num_on_line = board[11][line]
                    to_line = min(line + 1 - num_on_line, num_tiles)
                    to_floor = num_tiles - to_line
                if not best_move:
                    best_move = move
                    best_to_line = to_line
                    best_to_floor = to_floor
                elif to_line > best_to_line:
                    best_move = move
                    best_to_line = to_line
                    best_to_floor = to_floor
                elif to_line == best_to_line and to_floor < best_to_floor:
                    best_move = move
                    best_to_line = to_line
                    best_to_floor = to_floor
        return best_move
