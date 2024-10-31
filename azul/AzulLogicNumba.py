from .AzulLogic import np_factory_symmetries
import numpy as np
from numba import njit
import numba

############################## BOARD DESCRIPTION ##############################
# Board is described by a 23x6 array
# Colours are always in this order: 0: Blue, 1: Yellow, 2: Red, 3: Black, 4: White
# Here is the description of each line of the board. For readibility, we defined
# "shortcuts" that actually are views (numpy name) of overal board.
##### Index  Shortcut                Meaning
#####   0    self.scores             P0 score, P1 score, Round num, 0, 0, 0
#####  1-2   self.bag                Tiles in bag for each colour, 0
#####  2-3   self.discards           Tiles in discard pile for each colour, 0
#####  3-4   self.centre             Tiles of each colour in centre, (6th element is first player token)
#####  4-9   self.factories          Tiles of each colour in each factory, 0
#####  9     self.player_colours     Row colours of P0, -1 for empty, (6th element is first player token)
#####  10        =                   Row colours of P1, -1 for empty, (6th element is first player token)
#####  11    self.player_row_numbers Number of tiles on each row for P0, -1 for empty
#####  12        =                   Number of tiles on each row for P1, -1 for empty
#####  13-17 self.player_walls       P0 wall, 0 or 1 whether tile exists or not, 0
#####  18-22     =                   P1 wall, 0 or 1 whether tile exists or not, 0
# Any line follow by a 0 (or n 0s) means the last (or last n) columns is always 0


############################## ACTION DESCRIPTION #############################
# There are 180 actions, 6 choices of factory/centre to chose from, 5 choices
# of colour and 6 choice of destination line.
# 6*5*6 = 180
# To get action number do factory * 30 + colour * 6 + line
# Factory = {Centre: 0, Factory 1: 1, ..., Factory 5: 5}
# Colour = {Blue: 0, Yellow: 1, Red: 2, Black: 3, White: 4}
# Line = {Line 1: 0, Line 2: 1, ... Line 5: 4, Floor: 5}
# To further demonstrate:
##### Index  Meaning
#####   0    Centre, Blue, Line 1
#####   1    Centre, Blue, Line 2
#####  ...
#####   5    Centre, Blue, Floor
#####   6    Centre, Yellow, Line 1
#####  ...
#####   29   Centre, White, Floor
#####   30   Factory 1, Blue, Line 1
#####  ...
#####   59   Factory 1, White, Floor
#####  ...
#####   179  Factory 5, White, Floor

@njit(cache=True, fastmath=True, nogil=True)
def observation_size(num_players):
    return (23, 6)

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
    return 180

@njit(cache=True, fastmath=True, nogil=True)
def my_random_choice(prob):
    result = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
    return result

spec = [
    ('state' , numba.int8[:,:]),
    ('scores' , numba.int8[:,:]),
    ('bag' , numba.int8[:,:]),
    ('discards' , numba.int8[:,:]),
    ('centre' , numba.int8[:,:]),
    ('factories' , numba.int8[:,:]),
    ('player_colours' , numba.int8[:,:]),
    ('player_row_numbers' , numba.int8[:,:]),
    ('player_walls' , numba.int8[:,:]),
]
@numba.experimental.jitclass(spec)
class Board():
    def __init__(self):
        self.state = np.zeros((23, 6), dtype=np.int8)

    def get_score(self, player):
        return self.scores[0, player]

    def init_game(self):
        self.copy_state(np.zeros((23, 6), dtype=np.int8), copy_or_not=False)

        self.bag[:] = np.array([[20]*5 + [0]], dtype=np.int8)
        self.player_colours[:] = np.array([[-1]*5 + [0]] * 2, dtype=np.int8)
        _ = self.setup_new_round(0)
        return

    def get_state(self):
        return self.state

    def valid_moves(self, player):
        centre_colours_available = self.centre
        factory_colours_available = np.array([factory[colour] > 0 for factory in self.factories for colour in range(5)]).reshape(5, 5)
        player_colours = self.player_colours[player]
        line_free = player_colours == -1
        line_free[5] = True
        player_row_numbers = self.player_row_numbers[player]
        line_not_full = player_row_numbers < (np.arange(6) + 1)
        result = np.zeros(180, dtype=np.bool_)

        for factory in range(6):
            if factory == 0:
                colour_available = centre_colours_available[0].astype(np.bool_)
            else:
                colour_available = factory_colours_available[factory - 1, :]
            for colour in range(5):
                line_correct_colour = player_colours == colour
                wall_colour_free = np.ones(6, dtype=np.bool_)
                for i in range(5):
                    row_idx = 5 * player + i
                    col_idx = (colour + i) % 5
                    wall_colour_free[i] = self.player_walls[row_idx, col_idx] == 0
                free_line_valid = np.logical_and(line_free, wall_colour_free)
                partial_line_valid = np.logical_and(line_correct_colour, line_not_full)
                valid_lines = np.logical_or(free_line_valid, partial_line_valid)
                result[factory * 30 + colour * 6: factory * 30 + (colour + 1) * 6] = np.logical_and(colour_available[colour], valid_lines)
        return result

    def make_move(self, move, player, random_seed):
        if move < 30:
            factory = self.centre[0]
        else:
            factory = self.factories[(move - 30) // 30]
        colour = (move % 30) // 6
        line = move % 6
        num_tiles = factory[colour]
        if line == 5:
            to_floor = num_tiles
        else:
            num_on_line = self.player_row_numbers[player][line]
            to_line = min(line + 1 - num_on_line, num_tiles)
            to_floor = num_tiles - to_line
            self.player_row_numbers[player, line] += to_line
            self.player_colours[player, line] = colour
        self.player_row_numbers[player, 5] += to_floor
        self.discards[0, colour] += to_floor
        factory[colour] = 0
        if move < 30:
            if factory[5] == 1:
                self.player_row_numbers[player][5] += 1
                self.player_colours[player][5] = 1
                factory[5] = 0
        else:
            self.centre += factory
            factory[:] = [0]*6
        if np.logical_and(np.all(self.factories == 0), np.all(self.centre[0, :5] == 0)):
            self.score_round()
            next_player = self.setup_new_round(random_seed)
            if self.check_game_over():
                self.score_bonuses()
        else:
            next_player = (player + 1) % 2
        return next_player

    def check_game_over(self):
        game_over = False
        for i in range(10):
            if np.all(self.player_walls[i, :5] == 1):
                game_over = True
                break
        return game_over

    def score_round(self):
        rows_complete = self.player_row_numbers[:, :5] == np.arange(1, 6)
        player, row_nums = np.where(rows_complete)
        colours = np.empty(len(player), dtype=np.int8)
        for idx in range(len(player)):
            colours[idx] = self.player_colours[player[idx], row_nums[idx]]
        columns = (colours + row_nums) % 5
        wall_indices = zip(player, row_nums, columns)
        for p, r, c in wall_indices:
            self.scores[0, p] += self.score_change(self.player_walls[p*5: 5 + p*5, :5], r, c)
            self.player_walls[5*p + r, c] = 1
        for i in range(len(colours)):
            self.discards[0, colours[i]] += row_nums[i]
        for i in range(len(player)):
            self.player_row_numbers[player[i], row_nums[i]] = 0
            self.player_colours[player[i], row_nums[i]] = -1
        discard_mapping = {0:0, 1:1, 2:2, 3:4, 4:6, 5:8, 6:11, 7:14}
        self.scores[0, 0] = max(self.scores[0, 0] - discard_mapping[min(self.player_row_numbers[0, 5], 7)], 0)
        self.scores[0, 1] = max(self.scores[0, 1] - discard_mapping[min(self.player_row_numbers[1, 5], 7)], 0)
        self.player_row_numbers[0, 5] = 0
        self.player_row_numbers[1, 5] = 0
        return

    def score_bonuses(self):
        for p in range(2):
            player_walls = self.player_walls[5*p : 5 + 5*p, :5]
            for row in player_walls:
                if np.all(row == 1):
                    self.scores[0, p] += 2
            for col in range(5):
                if np.all(player_walls[:, col] == 1):
                    self.scores[0, p] += 7
            num_diags = 0
            num_diags = 0
            for i in range(5):
                diagonal_check = True
                for j in range(5):
                    if player_walls[j, (j + i) % 5] != 1:
                        diagonal_check = False
                        break
                if diagonal_check:
                    num_diags += 1
            self.scores[0, p] += num_diags * 10
        return

    @staticmethod
    def count_consecutive_ones(array, index):
        count = 1
        left = index - 1
        right = index + 1
        while left >= 0 and array[left] == 1:
            count += 1
            left -= 1
        while right < len(array) and array[right] == 1:
            count += 1
            right += 1
        return count

    def score_change(self, grid, r, c):
        grid[r, c] = 1
        row_adjacent = (c > 0 and grid[r, c-1] == 1) or (c < grid.shape[1] - 1 and grid[r, c+1] == 1)
        col_adjacent = (r > 0 and grid[r-1, c] == 1) or (r < grid.shape[0] - 1 and grid[r+1, c] == 1)
        if not row_adjacent and not col_adjacent:
            return 1
        row_score = self.count_consecutive_ones(grid[r, :], c) if row_adjacent else 0
        col_score = self.count_consecutive_ones(grid[:, c], r) if col_adjacent else 0
        return row_score + col_score


    def setup_new_round(self, random_seed):
        for i in range(5):
            if np.sum(self.bag) < 4:
                tiles_to_add = 4 - np.sum(self.bag)
                self.factories[i] = self.bag.copy()
                self.bag[:] = self.discards.copy()
                self.discards[:] = np.array([[0]*6], dtype=np.int8)
                self.factories[i] += self.select_tiles_from_bag(tiles_to_add, random_seed)
            else:
                self.factories[i] = self.select_tiles_from_bag(4, random_seed)
        if self.player_colours[1, 5] == 1:
            next_player = 1
            self.player_colours[1, 5] = 0
        else:
            next_player = 0
            self.player_colours[0, 5] = 0
        self.scores[0, 2] += 1
        self.centre[0, 5] = 1
        return next_player


    def select_tiles_from_bag(self, num, random_seed):
        result = np.zeros(6, dtype=np.int8)
        for _ in range(num):
            if random_seed == 0:
                selected_idx = my_random_choice(self.bag[0]/self.bag.sum())
            else:
                seed = (self.bag[0, :5] * 2**np.arange(5)).sum()
                fake_tile_num = (4594591 * (random_seed + seed)) % self.bag.sum()
                selected_idx = np.searchsorted(np.cumsum(self.bag[0, :5]), fake_tile_num, side='right')
            result[selected_idx] += 1
            self.bag[0, selected_idx] -= 1
        return result


    def copy_state(self, state, copy_or_not):
        if self.state is state and not copy_or_not:
                return
        self.state = state.copy() if copy_or_not else state
        self.scores = self.state[0 : 1 , :] # 1
        self.bag = self.state[1 : 2 , :] # 1
        self.discards = self.state[2 : 3 , :] # 1
        self.centre = self.state[3 : 4 , :] # 1
        self.factories = self.state[4 : 9 , :] # 5
        self.player_colours = self.state[9 : 11 , :] # 2
        self.player_row_numbers = self.state[11 : 13 , :] # 2
        self.player_walls = self.state[13 : 23 , :]     # 10

    def check_end_game(self):
        if self.check_game_over():
            row_totals = np.zeros(2, dtype=np.int8)
            for p in range(2):
                player_walls = self.player_walls[5*p : 5 + 5*p, :5]
                for row in player_walls:
                    if np.all(row == 1):
                        row_totals[p] += 1

            # Determine the output based on scores and row_totals
            if (self.scores[0, 0] > self.scores[0, 1]) or (self.scores[0, 0] == self.scores[0, 1] and row_totals[0] > row_totals[1]):
                out = np.array([1.0, -1.0], dtype=np.float32)
            elif (self.scores[0, 1] > self.scores[0, 0]) or (self.scores[0, 0] == self.scores[0, 1] and row_totals[1] > row_totals[0]):
                out = np.array([-1.0, 1.0], dtype=np.float32)
            else:
                out = np.array([0.01, 0.01], dtype=np.float32)
        else:
            out = np.array([0.0, 0.0], dtype=np.float32)
        return out

    def swap_players(self, _player):
        self.scores[0, 0], self.scores[0, 1] = self.scores[0, 1], self.scores[0, 0]
        self.player_colours[0], self.player_colours[1] = self.player_colours[1].copy(), self.player_colours[0].copy()
        self.player_row_numbers[0], self.player_row_numbers[1] = self.player_row_numbers[1].copy(), self.player_row_numbers[0].copy()
        self.player_walls[:5], self.player_walls[5:] = self.player_walls[5:].copy(), self.player_walls[:5].copy()
        return

    def get_symmetries(self, policy, valid_actions):
            def permute_factories(permutation):
                factories_copy = self.factories.copy()
                for i in range(5):
                    self.factories[i, :] = factories_copy[permutation[i], :]
                return
            def permute_array(array, permutation):
                new_array = array.copy()
                for i, p in enumerate(permutation):
                    new_array[30*(i+1):30*(i+2)] = array[30*(p+1):30*(p+2)]
                return new_array

            symmetries = []
            for permutation in np_factory_symmetries:
                factories_backup = self.factories.copy()
                permute_factories(permutation)
                new_policy = permute_array(policy, permutation)
                new_valid_actions = permute_array(valid_actions, permutation)
                symmetries.append((self.state.copy(), new_policy, new_valid_actions))
                self.factories[:] = factories_backup
            return symmetries

    def get_round(self):
        return self.scores[0, 2]
