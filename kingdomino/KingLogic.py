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

#def _print_players(board):
#    n = 2
#    # NAMES
#    print(' '*19, end='')
#    for p in range(n):
#        print(f'Player {p}', end='')
#        if p < n-1:
#            print(f' '*26, end='')
#    print()

    # Discards
#    print(f'{Style.BRIGHT}Discards: {Style.RESET_ALL}   ', end='')
#    print(' '*19, end='')
#    for p in range(n):
#        print(f'{board.player_row_numbers[0, p]}', end='')
#        if p < n-1:
#            print(f' '*26, end='')
#    print()

    # LINES
#    print(' '*9, end='')
#    for p in range(n):
#        for noble in board.players_nobles[3*p:3*p+3]:
#            if noble[idx_points] > 0:
#                print(f'  < {Style.BRIGHT}{noble[idx_points]}{Style.RESET_ALL} >  ', end='')
#            else:
#                print(f'        ', end='')
#        print(f' '*10, end='')
#    print()

    # WALLS
#    print(f'{Style.BRIGHT}Walls: {Style.RESET_ALL}   ', end='')
#    for p in range(n):
#        for c in range(6):
#            my_gems  = board.players_gems[p][c]
#            print(f'{light_colors[c]} {my_gems} {Style.RESET_ALL} ', end='')
#        print(f' Σ{board.players_gems[p].sum():2}      ', end='')
#    print()

    # CARDS
    # print()
#    print(f'{Style.BRIGHT}Cards: {Style.RESET_ALL}  ', end='')
#    for p in range(n):
#        for c in range(5):
#            my_cards = board.players_cards[p][c]
#            print(f'{light_colors[c]} {my_cards} {Style.RESET_ALL} ', end='')
#        print(f'              ', end='')
#    print()

    # RESERVED
#    if board.players_reserved.sum() > 0:
#        print()
#        for line in range(5):
#            if line == 2:
#                print(f'{Style.BRIGHT}Reserve: {Style.RESET_ALL}', end='')
#            else:
#                print(' '*9, end='')
#            for p in range(n):
#                for r in range(3):
#                    reserved = board.players_reserved[6*p+2*r:6*p+2*r+2]
#                    if reserved[0].sum() != 0:
#                        _print_card_line(reserved, line, 2)
#                    else:
#                        print(f' '*10, end='')
#                print(f' '*4, end='')
#            print()

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
