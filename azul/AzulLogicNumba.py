from Stochastic import hashed_draw
import numpy as np
from numba import njit
import numba

# ############################ BOARD DESCRIPTION ##############################
# Board is described by a (9N+5)x6 array, N = number of players (2, 3 or 4),
# i.e. 23x6 for 2 players (same layout as the historical 2-player version),
# 32x6 for 3 players, 41x6 for 4 players.
# There are F = 2N+1 factories (5, 7 or 9).
# Colours are always in this order: 0: Blue, 1: Yellow, 2: Red, 3: Black, 4: White
# Here is the description of each line of the board. For readibility, we defined
# "shortcuts" that actually are views (numpy name) of overal board.
# ### Index          Shortcut                Meaning
# ###   0            self.scores             P0 score, ..., P(N-1) score, Round num, then 0s
# ###   1            self.bag                Tiles in bag for each colour, 0
# ###   2            self.discards           Tiles in discard pile (box lid) for each colour, 0
# ###   3            self.centre             Tiles of each colour in centre, (6th element is first player token)
# ###   4   .. 4+F-1 self.factories          Tiles of each colour in each factory, 0
# ###   4+F .. +N-1  self.player_colours     Row colours of each player, -1 for empty, (6th element is first player token)
# ###   4+F+N ..     self.player_row_numbers Number of tiles on each row of each player, (6th element is floor)
# ###   4+F+2N ..    self.player_walls       Wall of each player (5 rows each), 0 or 1 whether tile exists or not, 0
# Any line follow by a 0 (or n 0s) means the last (or last n) columns is always 0


# ############################ ACTION DESCRIPTION #############################
# There are (F+1)*30 actions: F+1 choices of factory/centre to chose from, 5
# choices of colour and 6 choice of destination line.
# 2 players: 6*5*6 = 180, 3 players: 8*5*6 = 240, 4 players: 10*5*6 = 300
# To get action number do factory * 30 + colour * 6 + line
# Factory = {Centre: 0, Factory 1: 1, ..., Factory F: F}
# Colour = {Blue: 0, Yellow: 1, Red: 2, Black: 3, White: 4}
# Line = {Line 1: 0, Line 2: 1, ... Line 5: 4, Floor: 5}
# To further demonstrate:
# ### Index  Meaning
# ###   0    Centre, Blue, Line 1
# ###   1    Centre, Blue, Line 2
# ###  ...
# ###   5    Centre, Blue, Floor
# ###   6    Centre, Yellow, Line 1
# ###  ...
# ###   29   Centre, White, Floor
# ###   30   Factory 1, Blue, Line 1
# ###  ...
# ###   59   Factory 1, White, Floor
# ###  ...
# ###   30F+29  Factory F, White, Floor

# Number of non-identity factory permutations added per training position. All F!
# permutations (120 with 2 players, 5040 with 3, 362880 with 4) would make each
# position many near-identical examples; a fresh random subset per call keeps the
# augmentation diversity across positions at a fraction of the cost.
N_RANDOM_SYMMETRIES = 9


@njit(cache=True, fastmath=True, nogil=True)
def num_factories(num_players):
    return 2 * num_players + 1


@njit(cache=True, fastmath=True, nogil=True)
def observation_size(num_players):
    return (9 * num_players + 5, 6)


@njit(cache=True, fastmath=True, nogil=True)
def action_size(num_players):
    return (num_factories(num_players) + 1) * 30


@njit(cache=True, fastmath=True, nogil=True)
def my_random_choice(prob):
    result = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
    return result


spec = [
    ('num_players', numba.int8),
    ('num_factories', numba.int8),
    ('state', numba.int8[:, :]),
    ('scores', numba.int8[:, :]),
    ('bag', numba.int8[:, :]),
    ('discards', numba.int8[:, :]),
    ('centre', numba.int8[:, :]),
    ('factories', numba.int8[:, :]),
    ('player_colours', numba.int8[:, :]),
    ('player_row_numbers', numba.int8[:, :]),
    ('player_walls', numba.int8[:, :]),
]


@numba.experimental.jitclass(spec)
class Board():
    def __init__(self, num_players):
        self.num_players = num_players
        self.num_factories = num_factories(num_players)
        self.state = np.zeros(observation_size(num_players), dtype=np.int8)

    def get_score(self, player):
        return self.scores[0, player]

    def init_game(self):
        self.copy_state(np.zeros(observation_size(self.num_players), dtype=np.int8), copy_or_not=False)

        self.bag[0, :5] = 20
        self.player_colours[:, :5] = -1
        _ = self.setup_new_round(0, 0)
        return

    def get_state(self):
        return self.state

    def valid_moves(self, player):
        n = self.num_players
        centre_colours_available = self.centre
        factory_colours_available = np.array([factory[colour] > 0 for factory in self.factories for colour in range(5)]).reshape(self.num_factories, 5)
        player_colours = self.player_colours[player]
        line_free = player_colours == -1
        line_free[5] = True
        player_row_numbers = self.player_row_numbers[player]
        line_not_full = player_row_numbers < (np.arange(6) + 1)
        result = np.zeros(action_size(n), dtype=np.bool_)

        for factory in range(self.num_factories + 1):
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
        next_player = (player + 1) % self.num_players
        if np.logical_and(np.all(self.factories == 0), np.all(self.centre[0, :5] == 0)):
            self.score_round()
            # Token holder starts next round. If nobody took it (centre never
            # picked from), play simply continues in turn order.
            next_player = self.setup_new_round(random_seed, next_player)
            if self.check_game_over():
                self.score_bonuses()
        return next_player

    def check_game_over(self):
        game_over = False
        for i in range(5 * self.num_players):
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
            # int8 state: saturate instead of wrapping to a negative score
            self.scores[0, p] = min(np.int64(self.scores[0, p]) + self.score_change(self.player_walls[p*5: 5 + p*5, :5], r, c), 127)
            self.player_walls[5*p + r, c] = 1
        for i in range(len(colours)):
            self.discards[0, colours[i]] += row_nums[i]
        for i in range(len(player)):
            self.player_row_numbers[player[i], row_nums[i]] = 0
            self.player_colours[player[i], row_nums[i]] = -1
        floor_penalty = (0, 1, 2, 4, 6, 8, 11, 14)
        for p in range(self.num_players):
            self.scores[0, p] = max(np.int64(self.scores[0, p]) - floor_penalty[min(np.int64(self.player_row_numbers[p, 5]), 7)], 0)
            self.player_row_numbers[p, 5] = 0
        return

    def score_bonuses(self):
        for p in range(self.num_players):
            bonus = 0
            player_walls = self.player_walls[5*p: 5 + 5*p, :5]
            for row in player_walls:
                if np.all(row == 1):
                    bonus += 2
            for col in range(5):
                if np.all(player_walls[:, col] == 1):
                    bonus += 7
            num_diags = 0
            for i in range(5):
                diagonal_check = True
                for j in range(5):
                    if player_walls[j, (j + i) % 5] != 1:
                        diagonal_check = False
                        break
                if diagonal_check:
                    num_diags += 1
            bonus += num_diags * 10
            self.scores[0, p] = min(np.int64(self.scores[0, p]) + bonus, 127)
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

    def setup_new_round(self, random_seed, default_next_player):
        # Fill each factory with 4 tiles. When the bag runs out, refill it from the
        # box lid and continue; if both are empty, the remaining factories stay
        # (partially) empty, as in the official rules.
        for i in range(self.num_factories):
            self.factories[i, :] = 0
            needed = 4
            while needed > 0:
                in_bag = np.int64(self.bag[0, :5].sum())
                if in_bag == 0:
                    if self.discards[0, :5].sum() == 0:
                        break
                    self.bag[0, :] = self.discards[0, :]
                    self.discards[0, :] = 0
                    in_bag = np.int64(self.bag[0, :5].sum())
                nb = min(needed, in_bag)
                self.factories[i] += self.select_tiles_from_bag(nb, random_seed)
                needed -= nb
        next_player = default_next_player
        for p in range(self.num_players):
            if self.player_colours[p, 5] == 1:
                next_player = p
                self.player_colours[p, 5] = 0
        self.scores[0, self.num_players] += 1
        self.centre[0, 5] = 1
        return next_player

    def select_tiles_from_bag(self, num, random_seed):
        result = np.zeros(6, dtype=np.int8)
        for _ in range(num):
            if random_seed == 0:
                selected_idx = my_random_choice(self.bag[0]/self.bag.sum())
            else:
                # Deterministic draw: pure function of (seed, exact bag composition).
                # Base-21 key is collision-free (each count is in 0..20) and changes
                # after every draw since the bag total strictly decreases.
                bag_key = np.int64(0)
                for c in range(5):
                    bag_key = bag_key * 21 + np.int64(self.bag[0, c])
                fake_tile_num = hashed_draw(random_seed, bag_key, int(self.bag[0, :5].sum()))
                selected_idx = np.searchsorted(np.cumsum(self.bag[0, :5]), fake_tile_num, side='right')
            result[selected_idx] += 1
            self.bag[0, selected_idx] -= 1
        return result

    def copy_state(self, state, copy_or_not):
        if self.state is state and not copy_or_not:
            return
        self.state = state.copy() if copy_or_not else state
        n, f = self.num_players, self.num_factories
        self.scores = self.state[0: 1, :]                                    # 1
        self.bag = self.state[1: 2, :]                                       # 1
        self.discards = self.state[2: 3, :]                                  # 1
        self.centre = self.state[3: 4, :]                                    # 1
        self.factories = self.state[4: 4+f, :]                               # F
        self.player_colours = self.state[4+f: 4+f+n, :]                      # N
        self.player_row_numbers = self.state[4+f+n: 4+f+2*n, :]              # N
        self.player_walls = self.state[4+f+2*n: 4+f+7*n, :]                  # 5N

    def check_end_game(self):
        n = self.num_players
        if not self.check_game_over():
            return np.zeros(n, dtype=np.float32)
        # Highest score wins; ties are broken by the number of complete rows,
        # remaining ties share the victory
        keys = np.zeros(n, dtype=np.int64)
        for p in range(n):
            nb_rows = 0
            for row in self.player_walls[5*p: 5 + 5*p, :5]:
                if np.all(row == 1):
                    nb_rows += 1
            keys[p] = 16 * np.int64(self.scores[0, p]) + nb_rows
        winners = keys == keys.max()
        several_winners = winners.sum() > 1
        out = np.full(n, -1.0, dtype=np.float32)
        for p in range(n):
            if winners[p]:
                out[p] = 0.01 if several_winners else 1.0
        return out

    def swap_players(self, nb_swaps):
        # Rotate players so that player 'nb_swaps' becomes player 0:
        # new[i] = old[(i + nb_swaps) % N], same convention as other games
        n = self.num_players
        scores_copy = self.scores[0, :n].copy()
        colours_copy = self.player_colours.copy()
        numbers_copy = self.player_row_numbers.copy()
        walls_copy = self.player_walls.copy()
        for i in range(n):
            j = (i + nb_swaps) % n
            self.scores[0, i] = scores_copy[j]
            self.player_colours[i, :] = colours_copy[j, :]
            self.player_row_numbers[i, :] = numbers_copy[j, :]
            self.player_walls[5*i: 5*i+5, :] = walls_copy[5*j: 5*j+5, :]
        return

    def get_symmetries(self, policy, valid_actions):
        f = self.num_factories

        def permute_factories(permutation):
            factories_copy = self.factories.copy()
            for i in range(f):
                self.factories[i, :] = factories_copy[permutation[i], :]
            return

        def permute_array(array, permutation):
            new_array = array.copy()
            for i, p in enumerate(permutation):
                new_array[30*(i+1):30*(i+2)] = array[30*(p+1):30*(p+2)]
            return new_array

        # Identity first, then N_RANDOM_SYMMETRIES distinct other permutations
        # of the factories, drawn uniformly (rejection of identity and repeats)
        symmetries = [(self.state.copy(), policy.copy(), valid_actions.copy())]
        identity = np.arange(f)
        chosen = np.zeros((N_RANDOM_SYMMETRIES, f), dtype=np.int64)
        nb_chosen = 0
        while nb_chosen < N_RANDOM_SYMMETRIES:
            permutation = np.random.permutation(f)
            is_new = not np.all(permutation == identity)
            for k in range(nb_chosen):
                if np.all(chosen[k] == permutation):
                    is_new = False
                    break
            if not is_new:
                continue
            chosen[nb_chosen] = permutation
            nb_chosen += 1

            factories_backup = self.factories.copy()
            permute_factories(permutation)
            new_policy = permute_array(policy, permutation)
            new_valid_actions = permute_array(valid_actions, permutation)
            symmetries.append((self.state.copy(), new_policy, new_valid_actions))
            self.factories[:] = factories_backup
        return symmetries

    def get_round(self):
        return self.scores[0, self.num_players]
