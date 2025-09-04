import numpy as np
import random

# from .AkropolisDisplay import print_board, move_to_str, directions_char
from .AkropolisDisplay import move_to_str
from .AkropolisGame import AkropolisGame

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

# class GreedyPlayer():
#     def __init__(self, game):
#         self.game = game

#     def play(self, board, nb_moves):
#         valids = self.game.getValidMoves(board, player=0)

#         best_action, best_diff_score = -1, -100
#         for i, v in enumerate(valids):
#             if v:
#                 temp_game = AkropolisGame()
#                 temp_game.board.copy_state(self.game.board.state, True)
#                 temp_game.getNextState(temp_game.board.state, player=0, action=i, random_seed=0)
#                 scores = [temp_game.getScore(temp_game.board.state, player=p) for p in range(2)]
#                 if (scores[0]-scores[1]) > best_diff_score:
#                     best_action, best_diff_score = i, scores[0]-scores[1]
#         return best_action


from .AkropolisConstants import *
from .AkropolisLogicNumba import PATTERNS, PATTERN_NEI, NEIGHBORS

# ===========================================================================
# Dynamic pyramid coordinates
# ===========================================================================

lvl1_tiles =  [
    {(4, 7), (5, 6), (5, 7)},
    {(7, 4), (6, 5), (6, 4)},
    {(4, 4), (5, 3), (5, 4)},
    {(7, 2), (6, 2), (6, 3)},
    {(4, 2), (4, 3), (5, 2)},
]

surrounded_hexes = { (5,3), (6,5), (5,6) }
 
def _compute_level_pyramid(leftest_hex, n_tiles):
    r0, q0 = leftest_hex
    tiles = []
    for i in range(n_tiles):
        if r0 % 2 == 0:
            tile = {(r0, q0), (r0-1, q0), (r0, q0+1)}
            r0, q0 = r0-1, q0+1
        else:
            tile = {(r0, q0), (r0+1, q0+1), (r0, q0+1)}
            r0, q0 = r0+1, q0+2
        tiles.append(tile)
    return tiles

from copy import deepcopy
# universe = [ [], [{tile0_lvl0}, {tile1_lvl0}, ..], [{tile0_lvl1}, {tile1_lvl1}, ..] ]
# all_universes = { 0: universe0, 1: universe1, ... }
def _compute_all(all_universes, current_universe, current_sc_idx, leftest_hex, n_tiles):
    # Add new level to current universe
    universe_copy = deepcopy(current_universe)
    new_level = _compute_level_pyramid(leftest_hex, n_tiles)
    universe_copy.append(new_level)

    if n_tiles <= 1:
        all_universes[current_sc_idx] = universe_copy
        return all_universes, current_sc_idx+1

    # Recurrence for upper level
    r, q = leftest_hex
    if r % 2 == 0:
        next_leftest_hexes = [(r-1, q), (r, q+1)]
    else:
        next_leftest_hexes = [(r+1, q+1), (r, q+1)]
    for next_leftest_hex in next_leftest_hexes:
        all_universes, current_sc_idx = _compute_all(all_universes, universe_copy, current_sc_idx, next_leftest_hex, n_tiles-1)

    return all_universes, current_sc_idx

def _compute_scoring_positions(all_universes):
    result = {}
    for i, universe in all_universes.items():
        scoring_positions_by_level = []
        for level in range(len(universe)-1):
            # Lister toutes les coordonnées de level qui ne sont pas listées dans level+1
            hex_in_cur_level   = set([c for t in universe[level] for c in t])
            hex_in_upper_level = set([c for t in universe[level+1] for c in t])
            scoring_positions = hex_in_cur_level - hex_in_upper_level
            scoring_positions_by_level.append(scoring_positions)
        # For last level, all positions are scoring positions
        scoring_positions_by_level.append({c for t in universe[-1] for c in t})
        result[i] = scoring_positions_by_level
    return result

all_universes, _ = _compute_all({}, [[], lvl1_tiles], 0, (6,2), 4)
all_sp = _compute_scoring_positions(all_universes)

# ===========================================================================

def _is_non_blue_plaza(t: int) -> bool:
    return t in {PLAZA_RED, PLAZA_YELLOW, PLAZA_PURPLE, PLAZA_GREEN}

def _is_non_blue_district(t: int) -> bool:
    return t in {DISTRICT_RED, DISTRICT_YELLOW, DISTRICT_PURPLE, DISTRICT_GREEN}

def _is_bd_or_q(t: int) -> bool:
    return t in {DISTRICT_BLUE, QUARRY}

def neigh_it(r: int, q: int):
    """
    Yield all board-bounded neighbours of the axial cell (r, q)
    using the odd-r offset convention.
    """
    dirs = DIRECTIONS_EVEN if (r & 1) == 0 else DIRECTIONS_ODD
    for dr, dq in dirs:
        nr, nq = r + dr, q + dq
        if 0 <= nr < CITY_SIZE and 0 <= nq < CITY_SIZE:
            yield nr, nq


def _would_create_new_tileslot(game, tile_coords, tile_coords_set, tile_descr):
    best_pattern_score = (0, 0)
    # Check if a new tile at coords would unlock a new tileslot at the upper level
    # and count how many Q/BD under this new tile. Method is brute force
    for candidate_idx in range(PATTERNS.shape[0]):
        candidate_coords_set = set([divmod(int(idx), CITY_SIZE) for idx in PATTERNS[candidate_idx]])
        common_coords = candidate_coords_set & tile_coords_set
        candidate_only_coords = candidate_coords_set - tile_coords_set

        # Check that candidate tile has either 1 or 2 common hex with the main tile
        if len(common_coords) == 0 or len(candidate_only_coords) == 0:
            continue

        # Check that new tileslot would be a constant level
        tile_height = [game.board.board_height[r, q, 0] for (r,q) in common_coords][0] + 1
        other_heights = [game.board.board_height[r, q, 0] for (r,q) in candidate_only_coords]
        if any(oh > tile_height for oh in other_heights):
            continue

        # Check that tileslot doesn't fall in the pyramid
        pyramid_coords_u0 = all_universes[0][tile_height+1]
        if not all(candidate_coords_set.isdisjoint(pyr_tile) for pyr_tile in pyramid_coords_u0):
            continue

        # Count number of Q et NBD under the candidate tile
        n_quarry = sum(game.board.board_descr[r, q, 0] == QUARRY for (r,q) in candidate_only_coords)
        n_bd     = sum(game.board.board_descr[r, q, 0] == DISTRICT_BLUE for (r,q) in candidate_only_coords)
        # add the ones from the new tile
        n_quarry += sum(1 for (r, q) in common_coords if tile_descr[tile_coords.index((r, q))] == QUARRY)
        n_bd     += sum(1 for (r, q) in common_coords if tile_descr[tile_coords.index((r, q))] == DISTRICT_BLUE)

        pattern_score = (n_quarry, n_bd)
        if pattern_score > best_pattern_score:
            best_pattern_score = pattern_score

    return best_pattern_score

def action_features_per_universe(game, action: int, universe_idx: int):
    tile_idx, pattern_idx = divmod(action, CITY_SIZE * CITY_SIZE * N_ORIENTS)
    tile_id = int(game.board.construction_site[tile_idx, 3])
    tile_descr = TILES_DATA[tile_id, :3]
    board_descr = game.board.board_descr[:, :, 0]

    # ==== Tile-specific features ====
    has_nbp = any(_is_non_blue_plaza(h) for h in tile_descr)
    n_nbd = sum(_is_non_blue_district(h) for h in tile_descr)
    is_free_tile = (tile_idx == 0)
    rule1a_priority = sum([{PLAZA_GREEN: 4, PLAZA_RED: 3, PLAZA_PURPLE: 2, PLAZA_YELLOW: 1}.get(h, 0) for h in tile_descr])
    # ================================

    coords = [divmod(int(idx), CITY_SIZE) for idx in PATTERNS[pattern_idx]]
    coord_set = frozenset(coords)
    level = game.board.board_height[coords[0][0], coords[0][1], 0]
    pyramid_coords_level = all_universes[universe_idx][level+1] # array of set of coords
    scoring_positions_level = all_sp[universe_idx][level+1] # set of coords
    scoring_positions = set(sp for level in range(5) for sp in all_sp[universe_idx][level])

    # ==== Position-specific features ====
    is_in_pyramid = (coord_set in pyramid_coords_level)
    is_out_pyramid = all(coord_set.isdisjoint(tile) for tile in pyramid_coords_level)
    index_in_pyramid_level = pyramid_coords_level.index(coord_set) if is_in_pyramid else None
    n_hex_on_sp = sum(1 for (r, q) in coords if (r, q) in scoring_positions_level)
    rightmost_priority_for_0sp = max([c[1] for c in coords]) if n_hex_on_sp == 0 else 0
    # ====================================

    coords_of_yd_on_sp = [(r, q) for h, (r, q) in zip(tile_descr, coords) if (r, q) in scoring_positions_level and h == DISTRICT_YELLOW]
    hex_type_on_sp = [h for h, (r, q) in zip(tile_descr, coords) if (r, q) in scoring_positions_level]
    n_pd_surrounded = sum(1 for (r, q) in coords for (rn, qn) in neigh_it(r, q) if board_descr[rn, qn] == DISTRICT_PURPLE) # Count twice if same PD surrounded by 2 hexes
    # n_rd_surrounded = sum(1 for (r, q) in coords for (rn, qn) in neigh_it(r, q) if board_descr[rn, qn] == DISTRICT_RED)
    n_q_under, n_bd_under = _would_create_new_tileslot(game, coords, coord_set, tile_descr) if is_out_pyramid and has_nbp else (0, 0)
    nbd_rotation_priority = sum([{DISTRICT_GREEN: 30, DISTRICT_RED: 10, DISTRICT_YELLOW: 3, DISTRICT_PURPLE: 1}.get(h, 0) for h in tile_descr])
    n_sp_priority = ([2, 3, 1, 0] if n_nbd == 1 else [0, 1, 3, 2])[n_hex_on_sp] # 1>0>2>3sp if 1 NBD, else 2>3>1>0sp

    # ==== Complex features ====
    cover_BD_and_Q_only = all(_is_bd_or_q(board_descr[r, q]) for (r, q) in coords)
    rule1b_priority = 30*n_pd_surrounded + 5*n_q_under + n_bd_under
    rule3a_priority = 100*n_sp_priority + nbd_rotation_priority
    has_adjacent_yd_on_sp = any((rn, qn) in scoring_positions and board_descr[rn, qn] == DISTRICT_YELLOW for (r, q) in coords_of_yd_on_sp for (rn, qn) in neigh_it(r, q))
    has_nbp_on_sp = any(_is_non_blue_plaza(h) for h in hex_type_on_sp)
    n_nbd_on_sp = sum(1 for h in hex_type_on_sp if _is_non_blue_district(h))
    # ==========================

    buyable_tiles_id = [int(game.board.construction_site[tile_idx, 3]) for tile_idx in range(min(CONSTR_SITE_SIZE, game.board.stones[0]+1))]
    hex_coords_of_whole_pyramid = set([h for lvl in range(5) for tile in all_universes[universe_idx][lvl] for h in tile])
    hex_coords_of_whole_pyramid.add((7,5)) # Add the most southward hex from the initial tile

    # ==== Global features ====
    max_nbd_in_buyable_tiles = max([sum(1 for h in TILES_DATA[tid, :3] if _is_non_blue_district(h)) for tid in buyable_tiles_id])
    glob_hexes_out_of_pyramid = sum(game.board.board_height[r, q, 0] for r in range(CITY_SIZE) for q in range(CITY_SIZE) if (r, q) not in hex_coords_of_whole_pyramid)
    # =========================

    return {name: locals()[name] for name in (
        "has_nbp", "n_nbd", "is_free_tile", "rule1a_priority",
        "level", "rightmost_priority_for_0sp", "is_in_pyramid", "is_out_pyramid", "index_in_pyramid_level", "n_hex_on_sp",
        "cover_BD_and_Q_only", "rule1b_priority", "rule3a_priority", "has_adjacent_yd_on_sp", "has_nbp_on_sp", "n_nbd_on_sp",
        "max_nbd_in_buyable_tiles", "glob_hexes_out_of_pyramid",
    )}

def _fts_to_str(fts):
    result = f""
    result += ("NBP" if fts['has_nbp'] else "   ") + f"+{fts['n_nbd']}nbd "
    # result += "free " if fts['is_free_tile'] else "     "

    result += f"lvl{fts['level']}q{fts['rightmost_priority_for_0sp']} "
    result += f"in{fts['index_in_pyramid_level']} " if fts['is_in_pyramid'] else ("OUT " if fts['is_out_pyramid'] else "mix ")
    result += f"{fts['n_hex_on_sp']}sp=" + ("NBP" if fts['has_nbp_on_sp'] else "   ") + (f"+{fts['n_nbd_on_sp']}nbd " if fts['n_nbd_on_sp'] > 0 else "      ")

    result += "BDQ " if fts['cover_BD_and_Q_only'] else "    "
    result += f"1a={fts['rule1a_priority']} "
    result += f"1b={fts['rule1b_priority']:2} "
    result += f"3a={fts['rule3a_priority']:3} "
    result += "AYD " if fts['has_adjacent_yd_on_sp'] else "    "
    
    result += f"{fts['max_nbd_in_buyable_tiles']}, {fts['glob_hexes_out_of_pyramid']}"

    return result

def _check_universe(game, action: int, universe_idx: int, debug=False):
    pyramid_coords = all_universes[universe_idx]
    # highest_level = -1
    for level, tiles in enumerate(pyramid_coords):
        for tile in tiles:
            h_of_tile   = [game.board.board_height[r, q, 0] for (r, q) in tile]
            tID_of_tile = [game.board.board_tileID[r, q, 0] for (r, q) in tile]
            if any([h == level for h in h_of_tile]):
                # Check that the 3 hexes have either:
                #   the same height and same tileID
                #   a greater height
                tID_of_same_level = [tID for (h,tID) in zip(h_of_tile, tID_of_tile) if h == level]
                lower_level = [1 for h in h_of_tile if h < level]
                # highest_level = max(highest_level, level)
                if len(set(tID_of_same_level)) > 1 or len(lower_level) > 0:
                    if debug:
                        print(f'Fail general {level=} {tile=} {[game.board.board_height[r, q, 0] for (r, q) in tile]} {ids=}')
                        breakpoint()
                    return False

    # Check new tile
    _, pattern_idx = divmod(action, CITY_SIZE * CITY_SIZE * N_ORIENTS)
    coords = [divmod(int(idx), CITY_SIZE) for idx in PATTERNS[pattern_idx]]
    coord_set = frozenset(coords)
    level = game.board.board_height[coords[0][0], coords[0][1], 0] + 1
    if coord_set in pyramid_coords[level]:
        result = True
    else:
        result = all(coord_set.isdisjoint(t) for t in pyramid_coords[level])

    # if highest_level >= 3 and level >= 3 and result:
    #     print(f'{level=} {pyramid_coords[level]=} {coords=}')
    #     breakpoint()

    # if debug and not result and level > 2:
    #     print(f'Fail pour la tuile {level=} {pyramid_coords[level]=} {coords=}')
    #     breakpoint()

    return result

PRINT_DETAILS = True
class GreedyPlayer():
    def __init__(self, game):
        self.game = game
        self.nb_apply_rule_1b = 0

    def _categorize(self, board, debug=False):
        valids = self.game.getValidMoves(board, player=0)

        if PRINT_DETAILS:
            print()

        best_actions, best_category = [], -100
        for i, v in enumerate(valids):
            if not v:
                continue

            best_univ_idx, best_category_for_i, best_fts_for_i = [], -100, {}
            for universe_idx in range(8):
                if not _check_universe(self.game, i, universe_idx, debug=False):
                    continue

                fts = action_features_per_universe(self.game, i, universe_idx)
                category = 0

                if fts['has_adjacent_yd_on_sp']:
                    category = -10
                elif fts['has_nbp']:
                    if fts['is_in_pyramid'] and fts['has_nbp_on_sp'] and fts['level'] <= 1:
                        # Rule 1a
                        category = 50000 + 1000*(1-np.int32(fts['level'])) + 100*fts['n_nbd_on_sp'] + 10*fts['rule1a_priority'] + max(3-fts['index_in_pyramid_level'], 0)
                    elif fts['is_out_pyramid'] and fts['glob_hexes_out_of_pyramid'] < 3*3:
                        if fts['level'] >= 1 and fts['cover_BD_and_Q_only']:
                            # Rule 1b - level 2
                            category = 41000
                        elif fts['level'] == 0:
                            # Rule 1b - level 1
                            category = 40000 + fts['rule1b_priority']
                    elif fts['is_in_pyramid'] and fts['has_nbp_on_sp']:
                        if fts['n_nbd'] >= fts['max_nbd_in_buyable_tiles'] and fts['level'] >= 3:
                            # Rule 1c-1
                            category = 35000 + 1000*fts['n_nbd_on_sp'] + fts['rule1a_priority']
                        elif fts['level'] == 3:
                            # Rule 1c-2
                            category = 30000 + 1000*fts['n_nbd_on_sp'] + fts['rule1a_priority']
                elif fts['is_in_pyramid']:
                    if fts['n_nbd_on_sp'] >= 2 and fts['level'] >= 1:
                        # Rule 2
                        category = 29000
                    elif fts['is_free_tile'] and fts['level'] >= 1 and fts['n_nbd'] >= 1:
                        # Rule 3a
                        category = 10000 + 5000*fts['n_nbd_on_sp'] + 10*fts['rule3a_priority'] + fts['rightmost_priority_for_0sp']
                    elif fts['is_free_tile'] and fts['level'] >= 1 and fts['n_hex_on_sp'] == 0:
                        # Rule 3b-1
                        category = 7000 + fts['rightmost_priority_for_0sp']
                    elif fts['n_nbd'] >= 1 and fts['level'] >= 1:
                        # Rule 3b-2
                        category = 3000 + fts['rightmost_priority_for_0sp']
                    elif fts['is_free_tile']:
                        # Rule 3b-3
                        category = 1000 + fts['rightmost_priority_for_0sp']

                if category > best_category_for_i:
                    best_univ_idx, best_category_for_i, best_fts_for_i = universe_idx, category, fts


            _tid, _p = divmod(i, CITY_SIZE * CITY_SIZE * N_ORIENTS)
            _coords = [divmod(int(idx), CITY_SIZE) for idx in PATTERNS[_p]]
            if debug or (PRINT_DETAILS and best_category_for_i > best_category - 10000 and best_category_for_i > 0):
                print(f'{i:4} u{best_univ_idx}: {best_category_for_i:5} - {_tid} {_coords} {_fts_to_str(best_fts_for_i)}')
         
            if best_category_for_i > best_category:
                best_actions, best_category = [i], best_category_for_i
            elif best_category_for_i == best_category:
                best_actions.append(i)

        if PRINT_DETAILS:
            print(f'actions avec meilleur cat: {best_actions}, {best_category}')

        if 40000 <= best_category < 50000:
            self.nb_apply_rule_1b += 1

        return best_actions, best_category


    def play(self, board, nb_moves):
        best_actions, best_category = self._categorize(board, debug=False)
        
        # Discriminate best actions
        best_scores = []
        for a in best_actions:
            temp_game = AkropolisGame()
            temp_game.board.copy_state(self.game.board.state, True)
            temp_game.getNextState(temp_game.board.state, player=0, action=a, random_seed=0)
            score = temp_game.getScore(temp_game.board.state, player=0)
            best_scores.append(score)
        max_score = max(best_scores)
        best_actions = [a for a,s in zip(best_actions, best_scores) if s == max_score]
        if PRINT_DETAILS:
            print(f'actions avec meilleur sco: {best_actions}, {max_score}')
        action = random.choice(best_actions)
        if PRINT_DETAILS:
           print(f'action choisie: {action}')

        # if PRINT_DETAILS and 40000 <= best_category < 50000:
        #     breakpoint()

        return action

