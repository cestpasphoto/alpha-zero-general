import numpy as np
import random


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, _nb_moves):
        valids = self.game.getValidMoves(board, player=0)
        action = random.choices(range(self.game.getActionSize()), weights=valids.astype(np.int64), k=1)[0]
        return action


class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, nb_moves):
        valid = self.game.getValidMoves(board, 0)
        while True:
            factory = input("Factory?: (0 for centre): ")
            colour = input("Colour? : (0: Blue, 1: Yellow, 2: Red, 3: Black, 4: White): ")
            line = input("Line? : (6 for floor): ")
            input_move = int(factory) * 30 + int(colour) * 6 + int(line) - 1
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

    def play(self, board, _nb_moves):
        valids = self.game.getValidMoves(board, 0)
        n = self.game.getNumberOfPlayers()
        row_numbers_p0 = 4 + (2 * n + 1) + n  # see board description in AzulLogicNumba
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
                    num_on_line = board[row_numbers_p0][line]
                    to_line = min(line + 1 - num_on_line, num_tiles)
                    to_floor = num_tiles - to_line
                if best_move is None:
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
