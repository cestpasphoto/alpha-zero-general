from colorama import Style, Fore, Back
import numpy as np

factory_symmetries = [
 (0, 1, 2, 3, 4),
 (0, 1, 2, 4, 3),
 (0, 1, 3, 2, 4),
 (0, 1, 3, 4, 2),
 (0, 1, 4, 2, 3),
 (0, 1, 4, 3, 2),
 (0, 2, 1, 3, 4),
 (0, 2, 1, 4, 3),
 (0, 2, 3, 1, 4),
 (0, 2, 3, 4, 1),
 (0, 2, 4, 1, 3),
 (0, 2, 4, 3, 1),
 (0, 3, 1, 2, 4),
 (0, 3, 1, 4, 2),
 (0, 3, 2, 1, 4),
 (0, 3, 2, 4, 1),
 (0, 3, 4, 1, 2),
 (0, 3, 4, 2, 1),
 (0, 4, 1, 2, 3),
 (0, 4, 1, 3, 2),
 (0, 4, 2, 1, 3),
 (0, 4, 2, 3, 1),
 (0, 4, 3, 1, 2),
 (0, 4, 3, 2, 1),
 (1, 0, 2, 3, 4),
 (1, 0, 2, 4, 3),
 (1, 0, 3, 2, 4),
 (1, 0, 3, 4, 2),
 (1, 0, 4, 2, 3),
 (1, 0, 4, 3, 2),
 (1, 2, 0, 3, 4),
 (1, 2, 0, 4, 3),
 (1, 2, 3, 0, 4),
 (1, 2, 3, 4, 0),
 (1, 2, 4, 0, 3),
 (1, 2, 4, 3, 0),
 (1, 3, 0, 2, 4),
 (1, 3, 0, 4, 2),
 (1, 3, 2, 0, 4),
 (1, 3, 2, 4, 0),
 (1, 3, 4, 0, 2),
 (1, 3, 4, 2, 0),
 (1, 4, 0, 2, 3),
 (1, 4, 0, 3, 2),
 (1, 4, 2, 0, 3),
 (1, 4, 2, 3, 0),
 (1, 4, 3, 0, 2),
 (1, 4, 3, 2, 0),
 (2, 0, 1, 3, 4),
 (2, 0, 1, 4, 3),
 (2, 0, 3, 1, 4),
 (2, 0, 3, 4, 1),
 (2, 0, 4, 1, 3),
 (2, 0, 4, 3, 1),
 (2, 1, 0, 3, 4),
 (2, 1, 0, 4, 3),
 (2, 1, 3, 0, 4),
 (2, 1, 3, 4, 0),
 (2, 1, 4, 0, 3),
 (2, 1, 4, 3, 0),
 (2, 3, 0, 1, 4),
 (2, 3, 0, 4, 1),
 (2, 3, 1, 0, 4),
 (2, 3, 1, 4, 0),
 (2, 3, 4, 0, 1),
 (2, 3, 4, 1, 0),
 (2, 4, 0, 1, 3),
 (2, 4, 0, 3, 1),
 (2, 4, 1, 0, 3),
 (2, 4, 1, 3, 0),
 (2, 4, 3, 0, 1),
 (2, 4, 3, 1, 0),
 (3, 0, 1, 2, 4),
 (3, 0, 1, 4, 2),
 (3, 0, 2, 1, 4),
 (3, 0, 2, 4, 1),
 (3, 0, 4, 1, 2),
 (3, 0, 4, 2, 1),
 (3, 1, 0, 2, 4),
 (3, 1, 0, 4, 2),
 (3, 1, 2, 0, 4),
 (3, 1, 2, 4, 0),
 (3, 1, 4, 0, 2),
 (3, 1, 4, 2, 0),
 (3, 2, 0, 1, 4),
 (3, 2, 0, 4, 1),
 (3, 2, 1, 0, 4),
 (3, 2, 1, 4, 0),
 (3, 2, 4, 0, 1),
 (3, 2, 4, 1, 0),
 (3, 4, 0, 1, 2),
 (3, 4, 0, 2, 1),
 (3, 4, 1, 0, 2),
 (3, 4, 1, 2, 0),
 (3, 4, 2, 0, 1),
 (3, 4, 2, 1, 0),
 (4, 0, 1, 2, 3),
 (4, 0, 1, 3, 2),
 (4, 0, 2, 1, 3),
 (4, 0, 2, 3, 1),
 (4, 0, 3, 1, 2),
 (4, 0, 3, 2, 1),
 (4, 1, 0, 2, 3),
 (4, 1, 0, 3, 2),
 (4, 1, 2, 0, 3),
 (4, 1, 2, 3, 0),
 (4, 1, 3, 0, 2),
 (4, 1, 3, 2, 0),
 (4, 2, 0, 1, 3),
 (4, 2, 0, 3, 1),
 (4, 2, 1, 0, 3),
 (4, 2, 1, 3, 0),
 (4, 2, 3, 0, 1),
 (4, 2, 3, 1, 0),
 (4, 3, 0, 1, 2),
 (4, 3, 0, 2, 1),
 (4, 3, 1, 0, 2),
 (4, 3, 1, 2, 0),
 (4, 3, 2, 0, 1),
 (4, 3, 2, 1, 0)]

np_factory_symmetries = np.array(factory_symmetries, dtype=np.int8)

def move_to_str(move):
    colour_dict = {0: "B", 1: "Y", 2: "R", 3: "K", 4: "W"}
    factory_num = move // 30
    colour_num = (move % 30) // 6
    line_num = move % 6
    out = ('C' if factory_num == 0 else f'F{factory_num}') + colour_dict[colour_num] + str(line_num + 1)
    return out

import re

# Tile = 3 chars, coloured background + letter (readable without colours too)
COLOUR_LETTERS = 'BYRKW'
TILE_STYLE = [
    Back.BLUE + Fore.LIGHTWHITE_EX,          # Blue
    Back.LIGHTYELLOW_EX + Fore.BLACK,        # Yellow
    Back.RED + Fore.LIGHTWHITE_EX,           # Red
    Back.BLACK + Fore.LIGHTWHITE_EX,         # Black
    Back.LIGHTWHITE_EX + Fore.BLACK,         # White
]
SLOT_STYLE = [Fore.BLUE, Fore.YELLOW, Fore.RED, Fore.LIGHTBLACK_EX, Fore.WHITE]  # empty wall cell
TOKEN = Back.GREEN + Fore.BLACK + ' 1 ' + Style.RESET_ALL
EMPTY = Style.DIM + ' · ' + Style.RESET_ALL
FLOOR_PENALTIES = (1, 1, 2, 2, 2, 3, 3)
_ANSI = re.compile(r'\x1b\[[0-9;]*m')


def _tile(c):
    return f'{TILE_STYLE[c]}{Style.BRIGHT} {COLOUR_LETTERS[c]} {Style.RESET_ALL}'


def _wall_slot(c):
    return f'{SLOT_STYLE[c]}{Style.DIM} {COLOUR_LETTERS[c].lower()} {Style.RESET_ALL}'


def _pad(text, width):
    return text + ' ' * max(0, width - len(_ANSI.sub('', text)))


def _tiles_from_counts(counts):
    return ''.join(_tile(c) for c in range(5) for _ in range(int(counts[c])))


def _player_block(board, p):
    """Return the lines of one player's area: header, 5 pattern lines -> wall rows, floor."""
    score, has_token = int(board.get_score(p)), board.player_colours[p, 5] == 1
    lines = [f'{Style.BRIGHT}Player {p}{Style.RESET_ALL}  {Style.BRIGHT}{score:3d}{Style.RESET_ALL} pts'
             + (f'  {TOKEN}' if has_token else '')]
    wall = board.player_walls[5*p: 5*p + 5, :5]
    for r in range(5):
        n, c = int(board.player_row_numbers[p, r]), int(board.player_colours[p, r])
        pattern = EMPTY * (r + 1 - n) + (_tile(c) * n if n > 0 else '')
        row = ''.join(_tile((col - r) % 5) if wall[r, col] else _wall_slot((col - r) % 5) for col in range(5))
        lines.append(' ' * 3 * (4 - r) + pattern + ' ▸ ' + row)
    nb_floor = int(board.player_row_numbers[p, 5])
    penalty = sum(FLOOR_PENALTIES[:min(nb_floor, 7)])
    floor = ''
    for i in range(7):
        if i < nb_floor:
            floor += (TOKEN if (i == 0 and has_token) else f'{Back.LIGHTBLACK_EX}{Fore.BLACK} ■ {Style.RESET_ALL}')
        else:
            floor += f'{Style.DIM}-{FLOOR_PENALTIES[i]} {Style.RESET_ALL}'
    lines.append('floor ' + floor + (f' {Fore.LIGHTRED_EX}-{penalty}{Style.RESET_ALL}' if penalty else ''))
    return lines


def print_board(board):
    width = 38
    print(f'{Style.BRIGHT}══ Round {board.get_round()} ' + '═' * (2 * width - 10) + Style.RESET_ALL)
    # Factories, then centre; numbers match move_to_str (F1..F9, C)
    facs = []
    for i, factory in enumerate(board.factories):
        content = _tiles_from_counts(factory) if factory[:5].sum() > 0 else EMPTY
        facs.append(f'F{i+1} {content}')
    for start in range(0, len(facs), 5):  # at most 5 factories per line (<= 80 columns)
        print(' '.join(facs[start:start+5]))
    centre = board.centre[0]
    if centre[:5].sum() <= 20:
        centre_str = _tiles_from_counts(centre)
    else:  # up to 27 tiles with 4 players: show counts to stay within 80 columns
        centre_str = ' '.join(f'{_tile(c)}×{centre[c]}' for c in range(5) if centre[c] > 0)
    print('C  ' + (TOKEN if centre[5] else '') + centre_str)
    for name, pile in (('bag', board.bag[0]), ('lid', board.discards[0])):
        detail = ' '.join(f'{SLOT_STYLE[c]}{COLOUR_LETTERS[c]}{Style.RESET_ALL}{pile[c]:<2d}' for c in range(5))
        print(f'{Style.DIM}{name} {int(pile[:5].sum()):3d}{Style.RESET_ALL}  {detail}', end='    ')
    print()
    print()
    # Two player areas side by side per line
    for p in range(0, board.num_players, 2):
        left = _player_block(board, p)
        if p + 1 < board.num_players:
            right = _player_block(board, p + 1)
            for l, r in zip(left, right):
                print(_pad(l, width + 2) + '│ ' + r)
        else:
            for l in left:
                print(l)
        print()
