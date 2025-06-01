from .KingLogic import np_factory_symmetries
import numpy as np
from numba import njit
import numba



############################## BOARD DESCRIPTION ##############################
# Board is described by a 31x17 array
# Colours are always in this order: 0: Yellow (Fields), 1: Dark Green (Woods),
# 2: Blue (Sea), 3: Light Green (Grass), 4: Red (Desert), 5: Black (Cave)
# Here is the description of each line of the board. For readibility, we defined
# "shortcuts" that actually are views (numpy name) of overal board.
##### Index  Shortcut                Meaning
#####   0    self.scores             P0 score, P1 score, Round num, Current tile index, Current tile, 0, ..., 0
#####  1-3   self.bag                For each of the 33 tiles, how many are left in the bag, 0
#####  3-4   self.visible_tiles      T0 index, T0 owner, T1 index, T1 owner, ... T8 index, T8 owner, 0
#####  4-17  self.player_boards      P0 board, 7, 7, 7, 7
#####  17-30     =                   P1 board, 7, 7, 7, 7
# Any line follow by a 0 (or n 0s) means the last (or last n) columns is always 0


############################## ACTION DESCRIPTION #############################
# There are 681 actions
# First 4 are just choosing between the 4 tiles
# Next is discarding the tile
# Next 676 are placing the tile
# 169 possible locations for the top left of the tile and 4 orientations
# of colour and 6 choice of destination line.
# 6*5*6 = 180
# To get action number for tile placement on (x, y)
# 5 + (4 * (13 * x + y)) + tile_orientation
# tile_orintation = {Left top down: 0, Left top right: 1, Right top down: 2,
# Right top right: 3}
# To further demonstrate:
##### Index  Meaning
#####   0    Pick first tile
#####   1    Pick second tile
#####  ...
#####   3    Pick fourth tile
#####   4    Discard
#####   5    Place tile at (0, 0) with it's left side at top facing down
#####   6    Place tile at (0, 0) with it's left side at top facing right
#####   7    Place tile at (0, 0) with it's right side at top facing down
#####   8    Place tile at (0, 0) with it's right side at top facing right
#####   9    Place tile at (0, 1) with it's right side at top facing right
#####   10   Place tile at (0, 1) with it's right side at top facing right
#####   11   Place tile at (0, 1) with it's right side at top facing right
#####   12   Place tile at (0, 1) with it's right side at top facing right
#####  ...
#####   57    Place tile at (1, 0) with it's left side at top facing down
#####   58    Place tile at (1, 0) with it's left side at top facing right
#####   59    Place tile at (1, 0) with it's right side at top facing down
#####   60    Place tile at (1, 0) with it's right side at top facing right
#####  ...
#####   677   Place tile at (12, 12) with it's left side at top facing down
#####   678   Place tile at (12, 12) with it's left side at top facing right
#####   679   Place tile at (12, 12) with it's right side at top facing down
#####   680   Place tile at (12, 12) with it's right side at top facing right

@njit(cache=True, fastmath=True, nogil=True)
def observation_size(_num_players):
    return (31, 17)

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
    return 681

@njit(cache=True, fastmath=True, nogil=True)
def my_random_choice(prob):
    result = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
    return result

spec = [
    ('state' , numba.int8[:,:]),
    ('scores' , numba.int8[:,:]),
    ('bag' , numba.int8[:,:]),
    ('visible_tiles' , numba.int8[:,:]),
    ('player_boards' , numba.int8[:,:]),
]

tile_types = np.array([[0, 0, 0, 2],
                       [1, 1, 0, 4],
                       [2, 2, 0, 3],
                       [3, 3, 0, 2],
                       [4, 4, 0, 1],
                       [0, 1, 0, 1],
                       [0, 2, 0, 1],
                       [0, 3, 0, 1],
                       [0, 4, 0, 1],
                       [1, 2, 0, 1],
                       [1, 3, 0, 1],
                       [0, 1, 1, 1],
                       [0, 2, 1, 1],
                       [0, 3, 1, 1],
                       [0, 4, 1, 1],
                       [0, 5, 1, 1],
                       [1, 0, 1, 4],
                       [1, 2, 1, 1],
                       [1, 3, 1, 1],
                       [2, 0, 1, 2],
                       [2, 1, 1, 4],
                       [3, 0, 1, 1],
                       [3, 2, 1, 1],
                       [4, 0, 1, 1],
                       [4, 3, 1, 1],
                       [5, 0, 1, 1],
                       [3, 0, 2, 1],
                       [3, 2, 2, 1],
                       [4, 0, 2, 1],
                       [4, 3, 2, 1],
                       [5, 0, 2, 1],
                       [5, 4, 2, 2],
                       [5, 0, 3, 1],
                       [0, 0, 0, 0]],
                       dtype=np.int8)

starting_boards = np.zeros((26, 17), dtype=np.int8)
starting_boards = np.zeros((26, 17), dtype=np.int8)
starting_boards[:, 13:] = 7
starting_boards[6, 6] = 6
starting_boards[19, 6] = 6

@numba.experimental.jitclass(spec)
class Board():
    def __init__(self):
        self.state = np.zeros((31, 17), dtype=np.int8)

    def get_score(self, player):
        return self.scores[0, player]

    def init_game(self):
        self.copy_state(np.zeros((23, 6), dtype=np.int8), copy_or_not=False)

        self.bag[:] = tile_types[:, -1].reshape(2, 17)
        self.player_boards[:] = starting_boards.copy()
        _ = self.setup_new_round(0)
        return

    def get_state(self):
        return self.state

    def valid_moves(self, player):
        if 

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
        self.visible_tiles[8:16] = self.visible_tiles[0:8]
        self.visible_tiles[:8:2] = self.select_tiles_from_bag(4, random_seed)
        self.visible_tiles[1:9:2] = -1
        if self.visible_tiles[9] == 1:
            next_player = 1
        else:
            next_player = 0
        self.scores[0, 2] += 1
        return next_player


    def select_tiles_from_bag(self, num, random_seed):
        result = np.zeros(4, dtype=np.int8)
        for i in range(num):
            tile_counts = self.bag.reshape(1, 34)[0, :33]
            if random_seed == 0:
                selected_idx = my_random_choice(tile_counts/tile_counts.sum())
            else:
                seed = (tile_counts * 2**np.arange(33)).sum()
                fake_tile_num = (4594591 * (random_seed + seed)) % self.bag.sum()
                selected_idx = np.searchsorted(np.cumsum(tile_counts), fake_tile_num, side='right')
            result[i] = selected_idx
            self.bag[selected_idx // 17, selected_idx % 17] -= 1
        return result


    def copy_state(self, state, copy_or_not):
        if self.state is state and not copy_or_not:
                return
        self.state = state.copy() if copy_or_not else state
        self.scores = self.state[0 : 1 , :] # 1
        self.bag = self.state[1 : 3 , :] # 2
        self.visible_tiles = self.state[3 : 4 , :] # 1
        self.player_boards = self.state[4 : 30 , :] # 26

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
