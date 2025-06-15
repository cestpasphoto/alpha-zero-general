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
        move = -1
        while not move in np.where(valids)[0]:
            moveinput = input("Full describe move? (y/n): ")
            if moveinput == "y":
                xcoord = int(input("x coord relative to center?: "))
                ycoord = int(input("y coord relative to center?: "))
                rightup = int(input("Right side up? (0/1): "))
                pointingright = int(input("Pointing Right? (0/1): "))
                move = 5 + (4 * (13 * (6-ycoord) + (6+xcoord)) + (2*rightup + pointingright))
            elif moveinput == "n":
                move = int(input("Move: "))
        return move
