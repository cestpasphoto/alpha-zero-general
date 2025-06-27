from .KamisadoLogic import np_board_symmetries, reflection_perm, rotation_perm, tile_types
import numpy as np
from numba import njit
import numba


############################## BOARD DESCRIPTION ##############################
# Board is described by a 59x17 array
# Colours are always in this order: 0: Yellow (Fields), 1: Dark Green (Woods),
# 2: Blue (Sea), 3: Light Green (Grass), 4: Red (Desert), 5: Black (Cave)
# Here is the description of each line of the board. For readibility, we defined
# "shortcuts" that actually are views (numpy name) of overal board.
##### Index  Shortcut                Meaning
#####   0    self.scores             P0 score, P1 score, Round num, Tile num, Current tile index, Current tile, 0, ..., 0
#####  1-3   self.bag                For each of the 33 tiles, how many are left in the bag, 0
#####  3-4   self.visible_tiles      T0 index, T0 owner, T1 index, T1 owner, ... T8 index, T8 owner, 0
#####  4-5   self.tiles_to_chose     T1, T2, T3, T4, 0
#####  5-6   self.tiles_to_place     T1, T2, T3, T4, 0
#####  6-19  self.player_boards      P0 board, 0, 0, 0, 0
#####  19-32     =                   P1 board, 0, 0, 0, 0
#####  32-45 self.player_crowns      P0 crowns, 0, 0, 0, 0
#####  45-58     =                   P1 crowns, 0, 0, 0, 0
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
#####   9    Place tile at (0, 1) with it's left side at top facing down
#####   10   Place tile at (0, 1) with it's left side at top facing right
#####   11   Place tile at (0, 1) with it's right side at top facing down
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
    return (59, 17)

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
    ('tiles_to_chose' , numba.int8[:,:]),
    ('tiles_to_place' , numba.int8[:,:]),
    ('player_boards' , numba.int8[:,:]),
    ('player_crowns' , numba.int8[:,:]),
]


starting_boards = np.zeros((26, 17), dtype=np.int8)
starting_boards[:, :13] = -1
starting_boards[6, 6] = 6
starting_boards[19, 6] = 6

@numba.experimental.jitclass(spec)
class Board():
    def __init__(self):
        self.state = np.zeros((59, 17), dtype=np.int8)

    def get_score(self, player):
        return self.scores[0, player] + 256 * (self.scores[0, player] < 0)

    def init_game(self):
        self.copy_state(np.zeros((59, 17), dtype=np.int8), copy_or_not=False)
        self.bag[:] = np.ascontiguousarray(tile_types[:, -1]).reshape(2, 17)
        self.player_boards[:] = starting_boards.copy()
        self.visible_tiles[:] = np.array([[-1]*8 + [0]*9], dtype=np.int8)
        self.tiles_to_chose[:] = np.array([[-1]*16 + [0]], dtype=np.int8)
        _ = self.setup_new_round(0)
        return

    def get_state(self):
        return self.state

    @staticmethod
    def adjacent_to_value(arr, target):
        mask = (arr == target) | (arr == 6)
        adjacent = np.zeros_like(mask, dtype=np.bool_)
        adjacent[1:, :] |= mask[:-1, :]
        adjacent[:-1, :] |= mask[1:, :]
        adjacent[:, 1:] |= mask[:, :-1]
        adjacent[:, :-1] |= mask[:, 1:]
        return adjacent

    def valid_moves(self, player):
        result = np.zeros(681, dtype=np.bool_)
        if np.sum(self.visible_tiles[0, 9:17:2] == -1) > np.sum(self.visible_tiles[0, 1:9:2] != -1):
            result[:4] = self.visible_tiles[0, 1:9:2] == -1
        else:
            tile_board = self.player_boards[player*13:(player+1)*13, :13]
            empty_placement = tile_board == -1
            empty_placement_below = np.zeros_like(tile_board, dtype=np.bool_)
            empty_placement_right = np.zeros_like(tile_board, dtype=np.bool_)
            empty_placement_below[:-1, :] = empty_placement[1:, :]
            empty_placement_right[:, :-1] = empty_placement[:, 1:]
            tile_index = self.visible_tiles[0, 8:16:2][self.visible_tiles[0, 9:17:2] == player][0]
            tile_colours = tile_types[tile_index][:2]
            adjacent_to_c0 = self.adjacent_to_value(tile_board, tile_colours[0])
            if tile_colours[0] == tile_colours[1]:
                adjacent_to_c1 = adjacent_to_c0
            else:
                adjacent_to_c1 = self.adjacent_to_value(tile_board, tile_colours[1])
            adjacent_to_c0_below = np.zeros_like(adjacent_to_c0, dtype=np.bool_)
            adjacent_to_c1_below = np.zeros_like(adjacent_to_c0, dtype=np.bool_)
            adjacent_to_c0_right = np.zeros_like(adjacent_to_c0, dtype=np.bool_)
            adjacent_to_c1_right = np.zeros_like(adjacent_to_c0, dtype=np.bool_)
            adjacent_to_c0_below[:-1, :] = adjacent_to_c0[1:, :]
            adjacent_to_c1_below[:-1, :] = adjacent_to_c1[1:, :]
            adjacent_to_c0_right[:, :-1] = adjacent_to_c0[:, 1:]
            adjacent_to_c1_right[:, :-1] = adjacent_to_c1[:, 1:]
            result[5::4] = (empty_placement & empty_placement_below & (adjacent_to_c0 | adjacent_to_c1_below)).reshape(169)
            result[6::4] = (empty_placement & empty_placement_right & (adjacent_to_c0 | adjacent_to_c1_right)).reshape(169)
            result[7::4] = (empty_placement & empty_placement_below & (adjacent_to_c1 | adjacent_to_c0_below)).reshape(169)
            result[8::4] = (empty_placement & empty_placement_right & (adjacent_to_c1 | adjacent_to_c0_right)).reshape(169)
        if np.sum(result) == 0:
            result[4] = True
        return result

    def calculate_score(self, player):
        tiles = self.player_boards[player*13:(player+1)*13, :13]
        crowns = self.player_crowns[player*13:(player+1)*13, :13]
        visited = np.zeros((13, 13), dtype=np.bool_)
        score = 0
        stack = [(0, 0)] * (13 * 13)
        directions = [(-1,0), (1,0), (0,-1), (0,1)]

        for i in range(13):
            for j in range(13):
                if visited[i, j]:
                    continue

                tile = tiles[i, j]
                area = 1
                stars = crowns[i, j]
                visited[i, j] = True

                stack_size = 1
                stack[0] = (i, j)

                while stack_size > 0:
                    x, y = stack[stack_size - 1]
                    stack_size -= 1
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 13 and 0 <= ny < 13:
                            if (not visited[nx, ny]) and (tiles[nx, ny] == tile):
                                visited[nx, ny] = True
                                area += 1
                                stars += crowns[nx, ny]
                                stack[stack_size] = (nx, ny)
                                stack_size += 1

                if stars > 0:
                    score += area * stars

        return score

    def make_move(self, move, player, random_seed):
        tile_num = self.scores[0, 3]
        if move < 4:
            self.visible_tiles[0, 1:9:2][move] = player
            if tile_num == 3:
                if self.check_game_over():
                    self.score_bonuses()
                    next_player = 0
                else:
                    next_player = self.setup_new_round(random_seed)
            else:
                self.scores[0, 3] += 1
                next_player = self.visible_tiles[0, 9+(tile_num+1)*2]
                if next_player == -1:
                    if (tile_num % 2) == 0:
                        next_player = (player + 1) % 2
                    else:
                        next_player = player
        else:
            self.visible_tiles[0, 8+tile_num*2:8+(tile_num+1)*2] = [-1, -1]
            self.tiles_to_place[0, tile_num*4:(tile_num + 1)*4] = [-1, -1, -1, -1]
            next_player = player
            if move > 4:
                tile_orientation = (move - 5) % 4
                x = ((move - 5) // 4) // 13
                y = ((move - 5) // 4) % 13
                if tile_orientation < 2:
                    self.player_boards[13*player + x, y] = self.scores[0, 5]
                    self.player_crowns[13*player + x, y] = self.scores[0, 7]
                else:
                    self.player_boards[13*player + x, y] = self.scores[0, 6]
                if tile_orientation == 0:
                    self.player_boards[13*player + x + 1, y] = self.scores[0, 6]
                    if x < 6:
                        self.player_boards[13*player + x + 7, :13] = -2
                    if x < 5:
                        self.player_boards[13*player + x + 8, :13] = -2
                    if x > 5:
                        self.player_boards[13*player + x - 6, :13] = -2
                    if x > 6:
                        self.player_boards[13*player + x - 7, :13] = -2
                    if y < 6:
                        self.player_boards[13*player:13*(player+1), y+7] = -2
                    if y > 6:
                        self.player_boards[13*player:13*(player+1), y-7] = -2
                elif tile_orientation == 1:
                    self.player_boards[13*player + x, y + 1] = self.scores[0, 6]
                    if x < 6:
                        self.player_boards[13*player + x + 7, :13] = -2
                    if x > 6:
                        self.player_boards[13*player + x - 7, :13] = -2
                    if y < 6:
                        self.player_boards[13*player:13*(player+1), y+7] = -2
                    if y < 5:
                        self.player_boards[13*player:13*(player+1), y+8] = -2
                    if y > 6:
                        self.player_boards[13*player:13*(player+1), y-7] = -2
                    if y > 5:
                        self.player_boards[13*player:13*(player+1), y-6] = -2
                    if y > 7:
                        self.player_boards[13*player:13*(player+1), y-7] = -2
                elif tile_orientation == 2:
                    self.player_boards[13*player + x + 1, y] = self.scores[0, 5]
                    self.player_crowns[13*player + x + 1, y] = self.scores[0, 7]
                    if x < 6:
                        self.player_boards[13*player + x + 7, :13] = -2
                    if x < 5:
                        self.player_boards[13*player + x + 8, :13] = -2
                    if x > 5:
                        self.player_boards[13*player + x - 6, :13] = -2
                    if x > 6:
                        self.player_boards[13*player + x - 7, :13] = -2
                    if y < 6:
                        self.player_boards[13*player:13*(player+1), y+7] = -2
                    if y > 6:
                        self.player_boards[13*player:13*(player+1), y-7] = -2
                elif tile_orientation == 3:
                    self.player_boards[13*player + x, y + 1] = self.scores[0, 5]
                    self.player_crowns[13*player + x, y + 1] = self.scores[0, 7]
                    if x < 6:
                        self.player_boards[13*player + x + 7, :13] = -2
                    if x > 6:
                        self.player_boards[13*player + x - 7, :13] = -2
                    if y < 6:
                        self.player_boards[13*player:13*(player+1), y+7] = -2
                    if y < 5:
                        self.player_boards[13*player:13*(player+1), y+8] = -2
                    if y > 6:
                        self.player_boards[13*player:13*(player+1), y-7] = -2
                    if y > 5:
                        self.player_boards[13*player:13*(player+1), y-6] = -2
                    if y > 7:
                        self.player_boards[13*player:13*(player+1), y-7] = -2
                self.scores[0, player] = self.calculate_score(player)
            if tile_num < 3:
                self.scores[0, 4] = self.visible_tiles[0, 8+(tile_num+1)*2]
                self.scores[0, 5:9] = tile_types[self.scores[0, 4]]
            if self.scores[0, 2] == 13:
                if self.check_game_over():
                    self.score_bonuses()
                else:
                    self.scores[0, 3] += 1
                    next_player = self.visible_tiles[0, 9+(tile_num+1)*2]
        return next_player

    def check_game_over(self):
        game_over = (np.sum(self.bag) == 0) & (self.visible_tiles[0, 15] == -1) & (self.visible_tiles[0, 0] == -2)
        return game_over


    def score_bonuses(self):
        def all_masked_less_than_zero(board, mask):
            for i in range(board.shape[0]):
                for j in range(board.shape[1]):
                    if mask[i, j]:
                        if board[i, j] > -1:
                            return False
            return True

        def all_masked_not_equal_minus_one(board, mask):
            for i in range(board.shape[0]):
                for j in range(board.shape[1]):
                    if mask[i, j]:
                        if board[i, j] == -1:
                            return False
            return True

        mask = np.ones((13, 13), dtype=np.bool_)
        mask[3:10, 3:10] = False
        for p in range(2):
            board = self.player_boards[13*p:13*(p+1), :13]
            if all_masked_less_than_zero(board, mask):
                self.scores[0, p] += 10
            if all_masked_not_equal_minus_one(board, ~mask):
                self.scores[0, p] += 5
        return

    def setup_new_round(self, random_seed):
        self.visible_tiles[0, 8:16] = self.visible_tiles[0, 0:8]
        self.tiles_to_place[:] = self.tiles_to_chose.copy()
        if np.sum(self.bag) > 0:
            self.visible_tiles[0, :8:2] = np.sort(self.select_tiles_from_bag(4, random_seed))
            self.visible_tiles[0, 1:9:2] = -1
            for i in range(4):
                self.tiles_to_chose[0, i*4:(i+1)*4] = tile_types[self.visible_tiles[0, i*2]]
        else:
            self.visible_tiles[0, :8] = -2
            self.tiles_to_chose[:] = np.array([[-1]*16 + [0]], dtype=np.int8)
        self.scores[0, 3] = 0
        if self.visible_tiles[0, 9] != -1:
            self.scores[0, 4] = self.visible_tiles[0, 8]
            self.scores[0, 5:9] = tile_types[self.scores[0, 4]]
        if self.visible_tiles[0, 9] == 1:
            next_player = 1
        else:
            next_player = 0
        self.scores[0, 2] += 1
        return next_player


    def select_tiles_from_bag(self, num, random_seed):
        result = np.zeros(4, dtype=np.int8)
        for i in range(num):
            tile_counts = np.ascontiguousarray(self.bag).reshape(34)[:33]
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
        self.tiles_to_chose = self.state[4 : 5 , :] # 1
        self.tiles_to_place = self.state[5 : 6 , :] # 1
        self.player_boards = self.state[6 : 32 , :] # 26
        self.player_crowns = self.state[32 : 58 , :] # 26


    @staticmethod
    def largest_connected_component_size(arr):
        n, m = arr.shape
        visited = np.zeros((n, m), dtype=np.bool_)
        max_size = 0

        stack_x = np.empty(n * m, dtype=np.int16)
        stack_y = np.empty(n * m, dtype=np.int16)

        for i in range(n):
            for j in range(m):
                if visited[i, j]:
                    continue

                val = arr[i, j]
                visited[i, j] = True
                if val < 0:
                    continue
                size = 1
                stack_ptr = 1
                stack_x[0] = i
                stack_y[0] = j

                while stack_ptr > 0:
                    stack_ptr -= 1
                    x = stack_x[stack_ptr]
                    y = stack_y[stack_ptr]

                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < n and 0 <= ny < m:
                            if (not visited[nx, ny]) and (arr[nx, ny] == val):
                                visited[nx, ny] = True
                                stack_x[stack_ptr] = nx
                                stack_y[stack_ptr] = ny
                                stack_ptr += 1
                                size += 1

                max_size = max(max_size, size)

        return max_size

    def check_end_game(self):
        if self.check_game_over():
            row_totals = np.zeros(2, dtype=np.int8)
            for p in range(2):
                player_board = self.player_boards[5*p : 5 + 5*p, :5]
                for row in player_board:
                    if np.all(row == 1):
                        row_totals[p] += 1
            p1score = self.scores[0, 0] + (256 * (self.scores[0, 0] < 0))
            p2score = self.scores[0, 1] + (256 * (self.scores[0, 1] < 0))
            if p1score > p2score:
                out = np.array([1.0, -1.0], dtype=np.float32)
            elif p2score > p1score:
                out = np.array([-1.0, 1.0], dtype=np.float32)
            else:
                p0_largest_ter = self.largest_connected_component_size(self.player_boards[:13, :13])
                p1_largest_ter = self.largest_connected_component_size(self.player_boards[13:26, :13])
                if p0_largest_ter > p1_largest_ter:
                    out = np.array([1.0, -1.0], dtype=np.float32)
                elif p1_largest_ter > p0_largest_ter:
                    out = np.array([-1.0, 1.0], dtype=np.float32)
                else:
                    p0_crowns = np.sum(self.player_crowns[:13, :13])
                    p1_crowns = np.sum(self.player_crowns[13:26, :13])
                    if p0_crowns > p1_crowns:
                        out = np.array([1.0, -1.0], dtype=np.float32)
                    elif p1_largest_ter > p0_largest_ter:
                        out = np.array([-1.0, 1.0], dtype=np.float32)
                    else:
                        out = np.array([0.01, 0.01], dtype=np.float32)
        else:
            out = np.array([0.0, 0.0], dtype=np.float32)
        return out

    def swap_players(self, _player):
        self.scores[0, 0], self.scores[0, 1] = self.scores[0, 1], self.scores[0, 0]
        self.player_boards[:13], self.player_boards[13:] = self.player_boards[13:].copy(), self.player_boards[:13].copy()
        self.player_crowns[:13], self.player_crowns[13:] = self.player_crowns[13:].copy(), self.player_crowns[:13].copy()
        mask_0 = self.visible_tiles[0, 1::2] == 0
        mask_1 = self.visible_tiles[0, 1::2] == 1
        self.visible_tiles[0, 1::2][mask_0] = 1
        self.visible_tiles[0, 1::2][mask_1] = 0
        return

    def get_symmetries(self, policy, valid_actions):
            def rotate_players_board(player):
                board_copy = self.player_boards[13*player:13*(player+1), :13].copy()
                crowns_copy = self.player_crowns[13*player:13*(player+1), :13].copy()
                self.player_boards[13*player:13*(player+1), :13] = np.rot90(board_copy)
                self.player_crowns[13*player:13*(player+1), :13] = np.rot90(crowns_copy)
                return
            def rotate_array(array):
                new_array = array[rotation_perm]
                return new_array
            def reflect_players_board(player):
                board_copy = self.player_boards[13*player:13*(player+1), :13].copy()
                crowns_copy = self.player_crowns[13*player:13*(player+1), :13].copy()
                self.player_boards[13*player:13*(player+1), :13] = np.fliplr(board_copy)
                self.player_crowns[13*player:13*(player+1), :13] = np.fliplr(crowns_copy)
                return
            def reflect_array(array):
                new_array = array[reflection_perm]
                return new_array

            symmetries = []
            for perm in np_board_symmetries:
                boards_backup = self.player_boards.copy()
                crowns_backup = self.player_crowns.copy()
                new_policy = policy.copy()
                new_valid_actions = valid_actions.copy()
                for _ in range(perm[0]):
                    rotate_players_board(0)
                    new_policy = rotate_array(new_policy)
                    new_valid_actions = rotate_array(new_valid_actions)
                for _ in range(perm[1]):
                    rotate_players_board(1)
                for _ in range(perm[2]):
                    reflect_players_board(0)
                    new_policy = reflect_array(new_policy)
                    new_valid_actions = reflect_array(new_valid_actions)
                for _ in range(perm[3]):
                    reflect_players_board(1)
                symmetries.append((self.state.copy(), new_policy, new_valid_actions))
                self.player_boards[:] = boards_backup
                self.player_crowns[:] = crowns_backup
            return symmetries

    def get_round(self):
        return self.scores[0, 2]
