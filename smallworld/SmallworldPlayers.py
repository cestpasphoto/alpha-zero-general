import numpy as np
import random

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

    def show_all_moves(self, valids):
        for action, v in enumerate(valids):
            if v:
                print(action, end=' ')
        print()

    def play(self, board, nb_moves):
        # print_board(self.game.board)
        valids = self.game.getValidMoves(board, 0)
        print()
        print('='*60, 'type your move, or + to get the list of moves')
        while True:
            input_move = input()
            if input_move == '+':
                self.show_all_moves(valids)
            else:
                try:
                    a = int(input_move)
                    if not valids[a]:
                        raise Exception('')
                    break
                except:
                    print('Invalid move:', input_move)
        return a


class GreedyPlayer():
    pass