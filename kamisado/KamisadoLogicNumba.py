from .KamisadoLogic import board_colours
import numpy as np
from numba import njit
import numba


############################## BOARD DESCRIPTION ##############################
# Board is described by a 9x8 array
# Colours are always in this order: 0: Brown 1: Green, 2: Red 3: Yellow
# 4: Pink, 5: Purple, 6: Blue, 7: Orange
# Second player pieces have 1 in front, eg 12 for second player red piece
# Here is the description of each line of the board. For readibility, we defined
# "shortcuts" that actually are views (numpy name) of overal board.
##### Index  Shortcut                Meaning
#####  0-8   self.board              Board
#####  8-9   self.info               Colour to move, 0, ..., 0
# Any line follow by a 0 (or n 0s) means the last (or last n) columns is always 0


############################## ACTION DESCRIPTION #############################
# There are 8 * 7 * 3 = 168 actions
# To get action number do piece_colour * 21 + direction * 7 + num_steps - 1
# Directions = {Left Diagonal: 0, Up: 1, Right Diagonal: 2}
# To further demonstrate:
##### Index  Meaning
#####   0    Brown, Left, 1
#####   1    Brown, Left, 2
#####  ...
#####   7    Brown, Up, 1
#####   8    Brown, Up, 2
#####  ...
#####   21   Green, Left, 1
#####   22   Green, Left, 2
#####  ...
#####   167  Orange, Right, 7

@njit(cache=True, fastmath=True, nogil=True)
def observation_size(_num_players):
    return (9, 8)

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
    return 168


spec = [
    ('state' , numba.int8[:,:]),
    ('board' , numba.int8[:,:]),
    ('info' , numba.int8[:,:]),
]


@numba.experimental.jitclass(spec)
class Board():
    def __init__(self):
        self.state = np.zeros((9, 8), dtype=np.int8)

    def init_game(self):
        self.copy_state(np.zeros((9, 8), dtype=np.int8), copy_or_not=False)
        self.board[:] = -np.ones((8, 8), dtype=np.int8)
        self.board[7, :] = np.arange(8)
        self.board[0, :] = np.arange(8)[::-1] + 10
        self.info[0, 0] = -1
        return

    def get_state(self):
        return self.state

    def valid_moves(self, player):
        result = np.zeros(168, dtype=np.bool_)
        colour_to_move = self.info[0, 0]
        if colour_to_move == -1:
            for colour in range(8):
                for direction in range(3):
                    for distance in range(1, 7):
                        if direction == 0:
                            if colour <= distance:
                                result[colour * 21 + direction * 7 + distance - 1] = True
                        if direction == 1:
                            result[colour * 21 + direction * 7 + distance - 1] = True
                        if direction == 1:
                            if 7 - colour <= distance:
                                result[colour * 21 + direction * 7 + distance - 1] = True
        else:
            token = 10 * player + colour_to_move
            position = np.argwhere(self.board == token)[0]
            for direction in range(3):
                for distance in range(1, 7):
                    player_directon = 2 * player - 1
                    new_placement = (position[0] + player_direction * distance, position[1] + player_directon * (direction - 1) * distance)
                    if 0 <= new_placement[0] <= 7:
                        if 0 <= new_placement[1] <= 7:
                            if self.board[new_placement] == -1:
                                result[colour_to_move * 21 + direction * 7 + distance - 1] = True
        return result


    def make_move(self, move, player, random_seed):
        colour = move // 21
        direction = (move % 21) // 7
        distance = (move % 7) + 1
        token = 10 * player + colour
        position = np.argwhere(self.board == token)[0]
        player_directon = 2 * player - 1
        new_placement = (position[0] + player_direction * distance, position[1] + player_directon * (direction - 1) * distance)
        self.board[tuple(position)] = -1
        self.board[new_placement] = token
        self.info[0, 0] = board_colours[new_placement]
        next_player = (player + 1) % 2
        return next_player

    def check_game_over(self):
        game_over = (np.any((self.board[0] > 0) & (self.board[0] < 10))) | (np.any(self.board[7] >= 10))
        return game_over


    def copy_state(self, state, copy_or_not):
        if self.state is state and not copy_or_not:
                return
        self.state = state.copy() if copy_or_not else state
        self.board = self.board[0 : 8 , :] # 8
        self.info = self.info[8 : 9 , :] # 1


    def check_end_game(self):
        if (np.any((self.board[0] > 0) & (self.board[0] < 10))):
            out = np.array([1.0, -1.0], dtype=np.float32)
        elif np.any(self.board[7] >= 10)
            out = np.array([-1.0, 1.0], dtype=np.float32)
        else:
            out = np.array([0.0, 0.0], dtype=np.float32)
        return out


    def swap_players(self, _player):
        self.board[:] = np.flip(self.board)
        return


    def get_symmetries(self, policy, valid_actions):
            symmetries = [(self.state.copy(), policy, valid_actions)]
            return symmetries
