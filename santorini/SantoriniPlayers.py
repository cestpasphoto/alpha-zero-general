import numpy as np
import random

from .SantoriniDisplay import print_board, move_to_str, directions_char

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, player=0):
        valids = self.game.getValidMoves(board, player)
        action = random.choices(range(self.game.getActionSize()), weights=valids.astype(np.int), k=1)[0]
        return action


class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def show_all_moves(self, valid):
        print('  ', end='')
        for worker in range(2):
            print(f'Move w{worker+1} '  , end='')
            for d in directions_char:
                print(f'{d}   ', end='')
            print('   ', end='')
        print()
        
        for build_i, build_direction in enumerate(directions_char):
            print(f'Build {build_direction}: ', end='')
            for worker in range(2):
                for move_i, move_direction in enumerate(directions_char):
                    action = build_i + 8*move_i + 8*8*worker
                    if valid[action]:
                        print(f'{action:3d} ', end='')
                    else:
                        print('    ', end='')
                print(' '*11, end='')
            print() 

    def play(self, board):
        # print_board(self.game.board)
        game_started = (np.abs(self.game.board[:,:,0]).sum() == 6) # check if all workers are set
        valid = self.game.getValidMoves(board, 0)
        print()
        print('='*80)
        if game_started:
            self.show_all_moves(valid)
        else:
            print('Type coordinates for your worker (y then x). For example type "0 0" for upper left corner')
        print('*'*80)
        while True:
            input_move = input()
            if input_move == '+':
                if game_started:
                    self.show_all_moves(valid)
            else:
                try:
                    if game_started:
                        a = int(input_move)
                    else:
                        coordinates = [int(x) for x in input_move.split()]
                        a = 5*coordinates[0] + coordinates[1]
                    if not valid[a]:
                        raise Exception('')
                    break
                except:
                    print('Invalid move:', input_move)
        return a


class GreedyPlayer():
    pass