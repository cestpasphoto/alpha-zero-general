from colorama import Style, Fore, Back
import numpy as np


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
new_y = new_y % 12
rotation_perm = np.arange(681, dtype=np.int16)
rotation_perm[5:] = 5 + (4 * (new_x * 13 + new_y)) + new_tile_orientation

new_tile_orientation = (tile_orientation + ((tile_orientation % 2) * 2)) % 4
new_y = 12 - y
new_y[(new_tile_orientation != tile_orientation)] -= 1
new_y = new_y % 12
reflection_perm = np.arange(681, dtype=np.int16)
rotation_perm[5:] = 5 + (4 * (x * 13 + new_y)) + new_tile_orientation

def move_to_str(move):
    if move < 4:
        string = f"C{move}"
    if move == 4:
        string = "D"
    if move > 4:
        tile_orientation = (move - 5) % 4
        x = ((move - 5) // 4) // 13
        y = ((move - 5) // 4) % 13
        orientation_dict = {0: "LTD", 1: "LTR", 2: "RTD", 3: "RTR"}
        string = orientation_dict[tile_orientation] + " @ (" + str(x) + ", " + str(y) + ")"
    return string

def print_board(board):
    print(board.state)
    return
