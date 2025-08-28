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


def action_features_per_universe(game, board_state, action: int, universe_idx: int):
    """Return `(PYR, nb_NBP, nb_NBD, nb_IMP, nb_IMPonI, NIV)`.

    – *PYR*   : 1 / 0 / ‑1  (in / out / mixed)
    – *nb_NBP*: non‑blue plazas in tile
    – *nb_NBD*: non‑blue districts in tile
    – *nb_IMP*: important hexes in tile
    – *nb_IMPonI*: important hexes on an « I » position
    - nbp_on_I: non-blue plaza on an I position
    - nb_HEXonI: hexes on an I position
    - all_nb_IMPonI: important hexes of whole board on an I position
    – *NIV*   : level before the tile is placed
    """

    tile_idx, pattern_idx = divmod(action, CITY_SIZE * CITY_SIZE * N_ORIENTS)
    coords = [divmod(int(idx), CITY_SIZE) for idx in PATTERNS[pattern_idx]]
    coord_set = frozenset(coords)

    heights = game.board.board_height[:, :, 0]
    NIV = int(min(heights[r, q] for r, q in coords))

    tile_id = int(game.board.construction_site[tile_idx, 3])
    tile_descr = TILES_DATA[tile_id, :3]

    nb_NBP = sum(_is_non_blue_plaza(t) for t in tile_descr)
    nb_NBD = sum(_is_non_blue_district(t) for t in tile_descr)
    nb_IMP = nb_NBP + nb_NBD

    # Get data from current universe
    pyramid_coords = all_universes[universe_idx]
    pyramid_coords_NIV = all_universes[universe_idx][NIV+1] # array of set of coords
    scoring_positions = all_sp[universe_idx]
    scoring_positions_NIV = all_sp[universe_idx][NIV+1] # set of coords

    imp_on_I = sum(
        1
        for t, (r, q) in zip(tile_descr, coords)
        if (r, q) in scoring_positions_NIV and (_is_non_blue_plaza(t) or _is_non_blue_district(t))
    )
    nbp_on_I = sum(
        1
        for t, (r, q) in zip(tile_descr, coords)
        if (r, q) in scoring_positions_NIV and (_is_non_blue_plaza(t))
    )
    gd_on_I = sum(
        1
        for t, (r, q) in zip(tile_descr, coords)
        if (r, q) in scoring_positions_NIV and t == DISTRICT_GREEN
    )
    rd_on_I = sum(
        1
        for t, (r, q) in zip(tile_descr, coords)
        if (r, q) in scoring_positions_NIV and t == DISTRICT_RED
        if (r, q) not in surrounded_hexes
    )
    pd_on_I = sum(
        1
        for t, (r, q) in zip(tile_descr, coords)
        if (r, q) in scoring_positions_NIV and t == DISTRICT_PURPLE
        if (r, q) in surrounded_hexes
    )
    yd_on_I = sum(
        1
        for t, (r, q) in zip(tile_descr, coords)
        if (r, q) in scoring_positions_NIV and t == DISTRICT_YELLOW
        if all(game.board.board_descr[r_, q_, 0] != DISTRICT_YELLOW for (r_, q_) in neigh_it(r, q))
    )
    bd_on_I = sum(
        1
        for t, (r, q) in zip(tile_descr, coords)
        if (r, q) in scoring_positions_NIV and t == DISTRICT_BLUE
    )
    q_on_I = sum(
        1
        for t, (r, q) in zip(tile_descr, coords)
        if (r, q) in scoring_positions_NIV and t == QUARRY
    )
    hex_on_I = sum(
        1
        for t, (r, q) in zip(tile_descr, coords)
        if (r, q) in scoring_positions_NIV
    )

    if coord_set in pyramid_coords_NIV:
        PYR = pyramid_coords_NIV.index(coord_set) + 1
    else:
        # Check overlap with any pyramid tile on this level
        overlap = any(coord_set & grp for grp in pyramid_coords_NIV)
        PYR = -1 if overlap else 0

    all_nb_IMPonI = imp_on_I
    for level, sp_on_level in enumerate(scoring_positions):
        for r, q in sp_on_level:
            if (r, q) in coord_set:
                continue
            t = game.board.board_descr[r, q, 0]
            h = game.board.board_height[r, q, 0]
            if t != EMPTY and h == level:
                if _is_non_blue_plaza(t) or _is_non_blue_district(t):
                    all_nb_IMPonI += 1

    plaza_position_score   = 100*nbp_on_I +  50*gd_on_I +  40*rd_on_I +  30*pd_on_I +  20*yd_on_I - 10*bd_on_I - 10*q_on_I + (5-PYR)
    scoring_position_score =  10*nbp_on_I + 500*gd_on_I + 400*rd_on_I + 200*pd_on_I + 300*yd_on_I -    bd_on_I -  5*q_on_I

    return {name: locals()[name] for name in (
        "PYR", "nb_NBP", "nb_NBD",
        "nb_IMP", "imp_on_I", "nbp_on_I", "hex_on_I", "all_nb_IMPonI",
        "NIV",
        "plaza_position_score", "scoring_position_score"
    )}
    
def _check_universe(game, board_state, action: int, universe_idx: int, debug=False):
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

def action_features(game, board_state, action: int):
    best_result = {
        "PYR": -1, "nb_NBP": 0, "nb_NBD" : 0,
        "nb_IMP": 0, "imp_on_I": 0, "nbp_on_I": 0, "hex_on_I": 0, "all_nb_IMPonI": 0,
        "NIV": 0,
        "plaza_position_score": -1000, "scoring_position_score": -1000,
    }
    # print('univ   ', end='')
    for universe_idx in range(8):
        if not _check_universe(game, board_state, action, universe_idx, debug=False):
            continue
        result = action_features_per_universe(game, board_state, action, universe_idx)
        # print(f'{result["hex_on_I"]} ', end='')
        if result['scoring_position_score'] > best_result['scoring_position_score']:
            # if best_result['hex_on_I'] >= 1:
            #     breakpoint()
            best_result = result

    # print()
    return best_result

def score_out_of_pyramid(game, board_state, action: int, verbose=False):
    # ---------- decode action & fetch coordinates ----------
    tile_idx, pattern_idx = divmod(action, CITY_SIZE * CITY_SIZE * N_ORIENTS)
    tile_coords = [divmod(int(idx), CITY_SIZE) for idx in PATTERNS[pattern_idx]]
    tile_coords_set = set(tile_coords)
    nei_coords = {divmod(int(idx), CITY_SIZE) for idx in PATTERN_NEI[pattern_idx, :]} - set(tile_coords)
    # if verbose:
    #     print(tile_coords, nei_coords)

    best_pattern_score = [0, 0, 0]
    # Check if setting this tile at this location would unlock a new tileslot at the upper level
    # Checking by brute force
    for candidate_idx in range(PATTERNS.shape[0]):
        candidate_coords_set = set([divmod(int(idx), CITY_SIZE) for idx in PATTERNS[candidate_idx]])
        common_coords = candidate_coords_set & tile_coords_set
        candidate_only_coords = candidate_coords_set - tile_coords_set

        # Si 1 ou 2 hex en commun avec la tuile posée
        if len(common_coords) == 0 or len(candidate_only_coords) == 0:
            continue

        # Si pas niveau supérieur sur les 2 autres hex
        tile_height = [game.board.board_height[r, q, 0] for (r,q) in common_coords][0] + 1
        other_heights = [game.board.board_height[r, q, 0] for (r,q) in candidate_only_coords]
        if any(oh > tile_height for oh in other_heights):
            continue

        # Si aucun dans la pyramide (univers 0)
        pyramid_coords_u0 = all_universes[0][tile_height+1]
        if not all(candidate_coords_set.isdisjoint(pyr_tile) for pyr_tile in pyramid_coords_u0):
            continue

        # compter le nombre de Q et NBD. renvoyer le max(10*Q+ NBD)
        n_quarry = sum(game.board.board_descr[r, q, 0] == QUARRY for (r,q) in candidate_only_coords)
        n_bd = sum(game.board.board_descr[r, q, 0] == DISTRICT_BLUE for (r,q) in candidate_only_coords)

        # add the ones from the new tile
        tile_id = int(game.board.construction_site[tile_idx, 3])
        for (r, q) in common_coords:
            i = tile_coords.index((r, q))
            descr = TILES_DATA[tile_id][i]
            if descr == QUARRY:
                n_quarry +=1
            elif descr == DISTRICT_BLUE:
                n_bd += 1
        pattern_score = [0, n_quarry, n_bd]
        if pattern_score > best_pattern_score:
            best_pattern_score = pattern_score

    # Check if surrounding any PD
    for (nei_r, nei_q) in nei_coords:
        if game.board.board_descr[nei_r, nei_q, 0] == DISTRICT_PURPLE:
            count = sum(1 for nn in NEIGHBORS[nei_r*CITY_SIZE+nei_q, :] if nn in PATTERNS[pattern_idx])
            best_pattern_score[0] += count
            # best_pattern_score[0] += [1 for neinei_idx in PATTERN_NEI[nei_r*CITY_SIZE+nei_q, :] if game.board.board_descr[neinei_idx//CITY_SIZE, neinei_idx%CITY_SIZE] != EMPTY ]

    return best_pattern_score

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

            category = 0
            features = action_features(self.game, board, i)
            is_cheapest_move = (i // (CITY_AREA * N_ORIENTS) == 0)

            # Take order within level only if level == 1, otherwise don't care
            PYR_ORDER = (5-features['PYR']) if features['NIV'] <= 1 else 0

            # Plaza position
            if features['PYR'] >= 1 and features['nbp_on_I'] >= 1 and features['NIV'] <= 1:
                # -NIV, nb_IMP_on_I, nb_HEX_on_I, PYR_ORDER
                category = 50000 + 10000*(1-np.int32(features['NIV'])) + features['plaza_position_score']
            # Plaza out of pyramid
            elif features['PYR'] == 0 and features['nb_NBP'] >= 1:
                if features['NIV'] >= 1:
                     # allow only if covering nothing else than quarry and bd
                    tile_idx, pattern_idx = divmod(i, CITY_SIZE * CITY_SIZE * N_ORIENTS)
                    coords = [divmod(int(idx), CITY_SIZE) for idx in PATTERNS[pattern_idx]]
                    type_of_hexes_below = [self.game.board.board_descr[r, q, 0] for (r, q) in coords]
                    if set(type_of_hexes_below) <= {QUARRY, DISTRICT_BLUE}:
                        category = 41000
                elif self.nb_apply_rule_1b < 3:
                    score = score_out_of_pyramid(self.game, board, i)
                    category = 40000 + 100*score[0] + 10*score[1] + score[2]
            elif features['PYR'] >= 1 and features['nbp_on_I'] >= 1 and features['NIV'] == 3 and self.nb_apply_rule_1b >= 3:
                # -NIV, nb_IMP_on_I, nb_HEX_on_I, PYR_ORDER
                category = 30000 + features['plaza_position_score']
            # 2 NBD and 2 scoring position
            elif features['PYR'] >= 1 and features['nb_NBD'] >= 2 and features['imp_on_I'] >= 2:
                category = 20000 + 1000*(5-np.int32(features['NIV'])) + (features['scoring_position_score'] if features['NIV'] >= 1 else features['plaza_position_score'])
            # cheapest
            elif features['PYR'] >= 1 and is_cheapest_move:
                if features['nb_NBD'] >= 1:
                    category = 10000 + 1000*np.int32(features['NIV']) + 100*features['imp_on_I'] + (features['scoring_position_score'] if features['NIV'] >= 1 else features['plaza_position_score'])
                elif features['hex_on_I'] == 0:
                    category = 7000 + 100*np.int32(features['NIV'])
            elif features['PYR'] >= 1 and features['nb_NBD'] >= 1:
                category = 3000 + 100*np.int32(features['NIV'])       
            elif features['PYR'] >= 1 and is_cheapest_move:
                category = 100*np.int32(features['NIV']) 

            _tid, _p = divmod(i, CITY_SIZE * CITY_SIZE * N_ORIENTS)
            _coords = [divmod(int(idx), CITY_SIZE) for idx in PATTERNS[_p]]
            if debug or (PRINT_DETAILS and category > best_category - 10000 and category > 0):
                print(f'{i:4}: {category:5} - {_tid} {_coords} - {features}')
         
            if category > best_category:
                best_actions, best_category = [i], category
            elif category == best_category:
                best_actions.append(i)

        if PRINT_DETAILS:
            print(f'actions avec meilleur cat: {best_actions}, {best_category}')

        if 40000 <= best_category < 41000:
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

        # if PRINT_DETAILS and 100 <= best_category < 40000:
        #     breakpoint()

        return action

