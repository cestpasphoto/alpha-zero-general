import sys
from Game import Game
from .AzulLogic import print_board, move_to_str
from .AzulLogicNumba import Board, observation_size, action_size
import numpy as np

sys.path.append('..')

NUMBER_PLAYERS = 2


class AzulGame(Game):
    def __init__(self):
        super().__init__()
        self.num_players = NUMBER_PLAYERS
        self.board = Board()

    def getInitBoard(self):
        self.board.init_game()
        return self.board.get_state()

    def getBoardSize(self):
        return observation_size(self.num_players)

    def getActionSize(self):
        return action_size()

    def getNextState(self, board, player, action, random_seed=0):
        self.board.copy_state(board, True)
        next_player = self.board.make_move(action, player, random_seed)
        return (self.board.get_state(), next_player)

    def getValidMoves(self, board, player):
        self.board.copy_state(board, False)
        return self.board.valid_moves(player)

    def getGameEnded(self, board, next_player):
        self.board.copy_state(board, False)
        return self.board.check_end_game()

    def getScore(self, board, player):
        self.board.copy_state(board, False)
        return self.board.get_score(player)

    def getRound(self, board):
        self.board.copy_state(board, False)
        return self.board.get_round()

    def getCanonicalForm(self, board, player):
        if player == 0:
            return board
        self.board.copy_state(board, True)
        self.board.swap_players(player)
        return self.board.get_state()

    def getSymmetries(self, board, pi, valid_actions):
        self.board.copy_state(board, True)
        return self.board.get_symmetries(np.array(pi, dtype=np.float32), valid_actions)

    def stringRepresentation(self, board):
        return board.tobytes()

    def getNumberOfPlayers(self):
        return NUMBER_PLAYERS

    def moveToString(self, move, current_player):
        return move_to_str(move)

    def printBoard(self, numpy_board):
        board = Board()
        board.copy_state(numpy_board, False)
        print_board(board)
        return
