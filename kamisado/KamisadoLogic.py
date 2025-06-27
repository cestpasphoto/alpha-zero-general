from colorama import Style, Fore, Back
import numpy as np

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

board_symmetries = [
 (0, 0, 0, 0),
 (0, 0, 0, 1),
 (0, 0, 1, 0),
 (0, 0, 1, 1),
 (0, 1, 0, 0),
 (0, 1, 0, 1),
 (0, 1, 1, 0),
 (0, 1, 1, 1),
 (0, 2, 0, 0),
 (0, 2, 0, 1),
 (0, 2, 1, 0),
 (0, 2, 1, 1),
 (0, 3, 0, 0),
 (0, 3, 0, 1),
 (0, 3, 1, 0),
 (0, 3, 1, 1),
 (1, 0, 0, 0),
 (1, 0, 0, 1),
 (1, 0, 1, 0),
 (1, 0, 1, 1),
 (1, 1, 0, 0),
 (1, 1, 0, 1),
 (1, 1, 1, 0),
 (1, 1, 1, 1),
 (1, 2, 0, 0),
 (1, 2, 0, 1),
 (1, 2, 1, 0),
 (1, 2, 1, 1),
 (1, 3, 0, 0),
 (1, 3, 0, 1),
 (1, 3, 1, 0),
 (1, 3, 1, 1),
 (2, 0, 0, 0),
 (2, 0, 0, 1),
 (2, 0, 1, 0),
 (2, 0, 1, 1),
 (2, 1, 0, 0),
 (2, 1, 0, 1),
 (2, 1, 1, 0),
 (2, 1, 1, 1),
 (2, 2, 0, 0),
 (2, 2, 0, 1),
 (2, 2, 1, 0),
 (2, 2, 1, 1),
 (2, 3, 0, 0),
 (2, 3, 0, 1),
 (2, 3, 1, 0),
 (2, 3, 1, 1),
 (3, 0, 0, 0),
 (3, 0, 0, 1),
 (3, 0, 1, 0),
 (3, 0, 1, 1),
 (3, 1, 0, 0),
 (3, 1, 0, 1),
 (3, 1, 1, 0),
 (3, 1, 1, 1),
 (3, 2, 0, 0),
 (3, 2, 0, 1),
 (3, 2, 1, 0),
 (3, 2, 1, 1),
 (3, 3, 0, 0),
 (3, 3, 0, 1),
 (3, 3, 1, 0),
 (3, 3, 1, 1)]

np_board_symmetries = np.array(board_symmetries, dtype=np.int8)

values = np.arange(681, dtype=np.int16)
tile_orientation = (values[5:] - 5) % 4
x = ((values[5:] - 5) // 4) // 13
y = ((values[5:] - 5) // 4) % 13
new_tile_orientation = (tile_orientation - 1) % 4
new_x = y
new_y = 12 - x
new_y[(tile_orientation % 2) == 0] -= 1
new_y = new_y % 13
rotation_perm = np.arange(681, dtype=np.int16)
rotation_perm[5:] = 5 + (4 * (new_x * 13 + new_y)) + new_tile_orientation

new_tile_orientation = (tile_orientation + ((tile_orientation % 2) * 2)) % 4
new_y = 12 - y
new_y[(new_tile_orientation != tile_orientation)] -= 1
new_y = new_y % 13
reflection_perm = np.arange(681, dtype=np.int16)
reflection_perm[5:] = 5 + (4 * (x * 13 + new_y)) + new_tile_orientation

def move_to_str(move):
    if move < 4:
        string = f"C{move}"
    if move == 4:
        string = "D"
    if move > 4:
        tile_orientation = (move - 5) % 4
        y = 6 - ((move - 5) // 4) // 13
        x = ((move - 5) // 4) % 13 - 6
        orientation_dict = {0: "LTD", 1: "LTR", 2: "RTD", 3: "RTR"}
        string = orientation_dict[tile_orientation] + " @ (" + str(x) + ", " + str(y) + ")"
    return string

king_colours = [
    Back.LIGHTYELLOW_EX + Fore.BLACK,
    Back.GREEN + Fore.BLACK,
    Back.LIGHTBLUE_EX + Fore.WHITE,
    Back.LIGHTGREEN_EX + Fore.BLACK,
    Back.LIGHTRED_EX + Fore.BLACK,
    Back.BLACK  + Fore.WHITE,
    Back.MAGENTA + Fore.BLACK]


def _print_round_and_scores(board):
    print('='*20, f' Round {board.get_round()}    ', end='')
    for p in range(2):
        print(f'{Style.BRIGHT}P{p}{Style.RESET_ALL}: {board.get_score(p)} points  ', end='')
    print('='*20, Style.RESET_ALL)


def _print_visible_tiles(board):
    visible_tiles = board.visible_tiles[0]
    print()
    print("Tiles to choose:")
    print()
    row_lines = ['', '']  # top and bottom of each cell
    for i in range(4):
        tile_index = 2*i
        if visible_tiles[tile_index] != -2:
            tile = tile_types[visible_tiles[tile_index]]
            content = f'{tile[2]}' if tile[2] > 0 else ''
            cell_text = f'{content:^3}'  # centered in 3 spaces
            row_lines[0] += f'{Style.BRIGHT}{king_colours[tile[0]]}{cell_text}{Style.RESET_ALL}'
            row_lines[0] += f'{Style.BRIGHT}{king_colours[tile[1]]}   {Style.RESET_ALL}'
            row_lines[1] += f'{Style.BRIGHT}{king_colours[tile[0]]}   {Style.RESET_ALL}'  # blank under text to pad height
            row_lines[1] += f'{Style.BRIGHT}{king_colours[tile[1]]}   {Style.RESET_ALL}'  # blank under text to pad height
            if visible_tiles[tile_index + 1] != -1:
                row_lines[0] += f"{str(visible_tiles[tile_index + 1]).rjust(2) + ' ' * 3}"
            else:
                row_lines[0] += "     "
            row_lines[1] += "     "
    print(row_lines[0])
    print(row_lines[1])
    print()
    print("Tiles to place:")
    print()
    row_lines = ['', '']  # top and bottom of each cell
    for i in range(4):
        tile_index = 2*i + 8
        if visible_tiles[tile_index] != -1:
            tile = tile_types[visible_tiles[tile_index]]
            content = f'{tile[2]}' if tile[2] > 0 else ''
            cell_text = f'{content:^3}'  # centered in 3 spaces
            row_lines[0] += f'{Style.BRIGHT}{king_colours[tile[0]]}{cell_text}{Style.RESET_ALL}'
            row_lines[0] += f'{Style.BRIGHT}{king_colours[tile[1]]}   {Style.RESET_ALL}'
            row_lines[1] += f'{Style.BRIGHT}{king_colours[tile[0]]}   {Style.RESET_ALL}'  # blank under text to pad height
            row_lines[1] += f'{Style.BRIGHT}{king_colours[tile[1]]}   {Style.RESET_ALL}'  # blank under text to pad height
            if visible_tiles[tile_index + 1] != -1:
                row_lines[0] += f"{str(visible_tiles[tile_index + 1]).rjust(2) + ' ' * 3}"
            else:
                row_lines[0] += "     "
            row_lines[1] += "     "
    print(row_lines[0])
    print(row_lines[1])
    print()
    return


def _print_coloured_grid(grid, numbers_grid):
    empty_square = Back.LIGHTBLACK_EX
    for row_idx in range(grid.shape[0]):
        row_lines = ['', '']  # top and bottom of each cell

        for col_idx in range(grid.shape[1]):
            if grid[row_idx][col_idx] == -1:
                block = f'{empty_square}   {Style.RESET_ALL}'
                row_lines[0] += block
                row_lines[1] += block
            else:
                colour = king_colours[grid[row_idx][col_idx]]
                number = numbers_grid[row_idx][col_idx]
                content = f'{number}' if number > 0 else ''
                cell_text = f'{content:^3}'  # centered in 3 spaces
                row_lines[0] += f'{Style.BRIGHT}{colour}{cell_text}{Style.RESET_ALL}'
                row_lines[1] += f'{Style.BRIGHT}{colour}   {Style.RESET_ALL}'  # blank under text to pad height

        print(row_lines[0])
        print(row_lines[1])
    return# prints extra line to make box square-ish

def _print_boards(board):
    for i in range(2):
        print(f"Player {i}")
        print()
        arr = board.player_boards[13*i:13*(i + 1), :13]
        crowns = board.player_crowns[13*i:13*(i + 1), :13]
        crowns = crowns[~(arr == -2).all(axis=1)][:, ~(arr == -2).all(axis=0)]
        arr = arr[~(arr == -2).all(axis=1)][:, ~(arr == -2).all(axis=0)]
        _print_coloured_grid(arr, crowns)
        print()
    return

def print_board(board):
    _print_round_and_scores(board)
    _print_visible_tiles(board)
    _print_boards(board)
    return
