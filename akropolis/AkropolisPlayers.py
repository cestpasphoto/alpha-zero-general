import numpy as np
import random

# from .AkropolisDisplay import print_board, move_to_str, directions_char
from .AkropolisDisplay import move_to_str
# from .AkropolisConstants import _encode_action

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, nb_moves):
        valids = self.game.getValidMoves(board, player=0)
        # print(f'{valids.sum()}/{len(valids)} valid moves')
        action = random.choices(range(self.game.getActionSize()), weights=valids.astype(np.int16), k=1)[0]
        return action


class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def show_all_moves(self, valids):
        for i in np.flatnonzero(valids):
            print(f'{i} {move_to_str(i, 0)}')

    def play(self, board, nb_moves):
        valids = self.game.getValidMoves(board, 0)
        print()
        print('='*80)
        self.show_all_moves(valids)

        while True:
            input_move = input()
            if input_move == '+':
                if game_started:
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