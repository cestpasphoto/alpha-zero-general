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
from .AkropolisLogicNumba import PATTERNS, PATTERN_NEI  # noqa: E402

# ===========================================================================
# Hard‑coded pyramid coordinates
# ===========================================================================

_pyr_groups = {
    0: [
        {(4, 7), (5, 6), (5, 7)},
        {(7, 4), (6, 5), (6, 4)},
        {(4, 4), (5, 3), (5, 4)},
        {(7, 2), (6, 2), (6, 3)},
        {(4, 2), (4, 3), (5, 2)},
    ],
    1: [
        {(5, 7), (5, 6), (6, 7)},
        {(5, 5), (6, 5), (6, 6)},
        {(5, 4), (6, 4), (5, 3)},
        {(5, 2), (6, 2), (6, 3)},
    ],
    2: [
        {(5, 6), (6, 7), (6, 6)},
        {(5, 5), (6, 5), (5, 4)},
        {(6, 3), (5, 3), (6, 4)},
    ],
    3: [
        {(5, 5), (6, 6), (6, 5)},
        {(5, 3), (5, 4), (6, 4)},
    ],
    4: [
        {(5, 4), (6, 5), (6, 4)},
    ],
}

_pyr_coords = {
    lvl: set().union(*groups) for lvl, groups in _pyr_groups.items()
}

# Scoring coordinates from the pyramid
_pyr_I = {
    0: {(4, 7), (7, 4), (4, 4), (7, 2), (4, 2), (4, 3)},
    1: {(5, 7), (5, 2), (6, 2)},
    2: {(5, 6), (6, 7), (6, 3)},
    3: {(5, 3), (5, 5), (6, 6)},
    4: set(),
}

def _is_non_blue_plaza(t: int) -> bool:
    return t in {PLAZA_RED, PLAZA_YELLOW, PLAZA_PURPLE, PLAZA_GREEN}

def _is_non_blue_district(t: int) -> bool:
    return t in {DISTRICT_RED, DISTRICT_YELLOW, DISTRICT_PURPLE, DISTRICT_GREEN}

def action_features(game, board_state, action: int):
    """Return `(PYR, nb_NBP, nb_NBD, nb_IMP, nb_IMPonI, NIV)`.

    – *PYR*   : 1 / 0 / ‑1  (in / out / mixed)
    – *nb_NBP*: non‑blue plazas in tile
    – *nb_NBD*: non‑blue districts in tile
    – *nb_IMP*: important hexes in tile
    – *nb_IMPonI*: important hexes on an « I » position
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

    imp_on_I = sum(
        1
        for t, (r, q) in zip(tile_descr, coords)
        if (r, q) in _pyr_I.get(NIV, set()) and (_is_non_blue_plaza(t) or _is_non_blue_district(t))
    )
    nbp_on_I = sum(
        1
        for t, (r, q) in zip(tile_descr, coords)
        if (r, q) in _pyr_I.get(NIV, set()) and (_is_non_blue_plaza(t))
    )

    # --- PYR according to exact group membership --------------------------------
    groups_lvl = _pyr_groups.get(NIV, [])
    if coord_set in groups_lvl:
        PYR = groups_lvl.index(coord_set) + 1
    else:
        # Check overlap with any pyramid tile on this level
        overlap = any(coord_set & grp for grp in groups_lvl)
        PYR = -1 if overlap else 0

    return PYR, nb_NBP, nb_NBD, nb_IMP, imp_on_I, nbp_on_I, NIV


def tile_neighbor_counts(game, board_state, action: int):
    """Return `(n_quarry, n_blue, n_non_blue)`.

    Included hexes:
    1. The **three** hexes of the tile to be placed.
    2. All hexes listed in **`PATTERN_NEI[pattern_idx]`** (direct neighbours of
       the tile) **whose coordinate at the top occupied level is not part of
       the pyramid**.
    """

    # ---------- decode action & fetch coordinates ----------
    tile_idx, pattern_idx = divmod(action, CITY_SIZE * CITY_SIZE * N_ORIENTS)
    tile_coords = [divmod(int(idx), CITY_SIZE) for idx in PATTERNS[pattern_idx]]
    nei_coords = {divmod(int(idx), CITY_SIZE) for idx in PATTERN_NEI[pattern_idx, :]} - set(tile_coords)

    # ---------- counts from tile content ----------
    tile_id = int(game.board.construction_site[tile_idx, 3])
    tile_descr = TILES_DATA[tile_id, :3]

    n_quarry = sum(t == QUARRY for t in tile_descr)
    n_blue = sum(t == DISTRICT_BLUE for t in tile_descr)
    n_non_blue = sum(_is_non_blue_district(t) for t in tile_descr)

    # ---------- counts from relevant neighbours ----------
    for r, q in nei_coords:
        if not (0 <= r < CITY_SIZE and 0 <= q < CITY_SIZE):
            continue  # out of board (should not happen)
        h = game.board.board_height[r, q, 0]
        if h == 0:
            continue  # empty cell
        lvl = h - 1  # topmost occupied layer

        # skip neighbour if *that* (r,q,lvl) lies inside pyramid
        if (r, q) in _pyr_coords.get(lvl, set()):
            continue

        t = game.board.board_descr[r, q, 0]
        if t == QUARRY:
            n_quarry += 1
        elif t == DISTRICT_BLUE:
            n_blue += 1
        elif _is_non_blue_district(t):
            n_non_blue += 1

    return int(n_quarry), int(n_blue), int(n_non_blue)


class GreedyPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, nb_moves):
        valids = self.game.getValidMoves(board, player=0)

        # print()

        best_actions, best_category = [], -100
        for i, v in enumerate(valids):
            if not v:
                continue

            category = 0
            PYR, nb_NBP, nb_NBD, nb_IMP, nb_IMP_on_I, nb_NBP_on_I, NIV = action_features(self.game, board, i)
            is_cheapest_move = (i // (CITY_AREA * N_ORIENTS) == 0)

            if PYR >= 1 and nb_NBP >= 1:
                if nb_IMP_on_I >= 2 and nb_NBP_on_I >= 1:
                    if NIV == 0:
                        category = 4300 + (10-PYR)
                    elif NIV == 1:
                        category = 4200 + (10-PYR) 
                elif nb_NBP_on_I >= 1:
                    if NIV == 0:
                        category = 4100 + (10-PYR)
                    elif NIV == 1:
                        category = 4000 + (10-PYR)
            elif PYR == 0 and nb_NBP >= 1:
                if NIV == 1:
                    category = 3000
                elif NIV == 0:
                    n_quarry, n_blue, n_non_blue = tile_neighbor_counts(self.game, board, i)
                    if n_quarry >= 3:
                        category = 2500
                    elif n_quarry >= 2 and n_blue >= 1:
                        category = 2200
                    elif n_quarry >= 1 and n_blue >= 2:
                        category = 2100
                    elif n_quarry >= 1 and n_blue >= 1 and n_non_blue >= 1:
                        category = 2000
            elif PYR >= 1 and nb_NBD >= 2 and nb_IMP_on_I >= 1:
                category = 1000 + 100*np.int16(NIV) + 10*(np.int16(nb_IMP_on_I)-1) + (10-PYR)
            elif PYR >= 1 and is_cheapest_move:
                if nb_NBD == 1 and nb_IMP_on_I >= 1:
                    category = 500 + 100*np.int16(NIV) + (10-PYR)
                elif nb_IMP_on_I == 0:
                    category = 100 + 100*np.int16(NIV) + (10-PYR)

            _tid, _p = divmod(i, CITY_SIZE * CITY_SIZE * N_ORIENTS)
            _coords = [divmod(int(idx), CITY_SIZE) for idx in PATTERNS[_p]]
            # if category > 0:
            #     print(f'{i: 3}: {category: 4} - {_tid} {_coords} - {PYR=} {nb_NBP=} {nb_NBD=} {nb_IMP=} {nb_IMP_on_I=} {NIV=}')

            if category > best_category:
                best_actions, best_category = [i], category
            elif category == best_category:
                best_actions.append(i)

        # print(f'actions avec meilleur cat: {best_actions}, {best_category}')

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
        # print(f'actions avec meilleur sco: {best_actions}, {max_score}')
        action = random.choice(best_actions)
        # print(f'action choisie: {action}')

        # if best_category < 1000:
        #     breakpoint()

        return action
