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
    factory_dict = {0: "C", 1: "F1", 2: "F2", 3: "F3", 4: "F4", 5: "F5"}
    colour_dict = {0: "B", 1: "Y", 2: "R", 3: "K", 4: "W"}
    factory_num = move // 30
    colour_num = (move % 30) // 6
    line_num = move % 6
    out = factory_dict[factory_num] + colour_dict[colour_num] + str(line_num + 1)
    return out

azul_colours = [
    Back.LIGHTBLUE_EX + Fore.WHITE,
    Back.LIGHTYELLOW_EX + Fore.BLACK,
    Back.LIGHTRED_EX + Fore.BLACK,
    Back.BLACK  + Fore.WHITE,
    Back.LIGHTWHITE_EX + Fore.BLACK]

azul_first_player_colour = Back.LIGHTGREEN_EX + Fore.BLACK


def _print_round_and_scores(board):
    print('='*20, f' Round {board.get_round()}    ', end='')
    for p in range(2):
        print(f'{Style.BRIGHT}P{p}{Style.RESET_ALL}: {board.get_score(p)} points  ', end='')
    print('='*20, Style.RESET_ALL)

def _print_factories(board):
    print(f'{Style.BRIGHT}Factories:  {Style.RESET_ALL}', end='')
    for i, factory in enumerate(board.factories):
        if np.sum(factory) != 0:
            print(f'< F{i+1} ', end='')
            for j, color in enumerate(azul_colours):
                if factory[j] != 0:
                    print(f'{color} {factory[j]} {Style.RESET_ALL} ', end='')
            print(f'> ', end='')
    print(f'{Style.RESET_ALL}')

def _print_centre(board):
    print(f'{Style.BRIGHT}Centre:  {Style.RESET_ALL}', end='')
    centre = board.centre
    for j, color in enumerate(azul_colours):
        if centre[0, j] != 0:
            print(f'{color} {centre[0, j]} {Style.RESET_ALL} ', end='')
    if centre[0, 5] > 0:
        print(f'{azul_first_player_colour} {1} {Style.RESET_ALL} ', end='')
    print(f'{Style.RESET_ALL}')

def _print_coloured_grid(grid):
    empty_square = Back.LIGHTBLACK_EX
    for row_idx in range(5):
        for col_idx in range(5):
            colour = azul_colours[(col_idx - row_idx) % 5]
            if grid[row_idx][col_idx] == 0:
                print(f'{empty_square}  {Style.RESET_ALL}', end='')
            else:
                print(f'{Style.BRIGHT}{colour}  {Style.RESET_ALL}', end='')
        print()
    return

def _print_coloured_rows(colored_counts, color_indices):
    for row_size in range(1, 6):
        colored_count = colored_counts[row_size - 1]
        color_index = color_indices[row_size - 1]
        for _ in range(colored_count):
            print(f'{azul_colours[color_index]}   {Style.RESET_ALL}', end='')
        for _ in range(row_size - colored_count):
            print(f'{Back.LIGHTBLACK_EX}   {Style.RESET_ALL}', end='')
        print()

def _print_players(board):
    for i in range(2):
        print(f"Player {i}")
        print(f"Dicards: {board.player_row_numbers[i, 5]}")
        print(f"First Player Token: {board.player_colours[i, 5]}")
        print()
        _print_coloured_rows(board.player_row_numbers[i, :5], board.player_colours[i, :5])
        print()
        _print_coloured_grid(board.player_walls[5*i: 5*(i+1), :5])
        print()
    return


def print_board(board):
    _print_round_and_scores(board)
    _print_factories(board)
    _print_centre(board)
    print()
    _print_players(board)
    return
