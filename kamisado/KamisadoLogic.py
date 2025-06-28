from rich.console import Console
from rich.style import Style
import numpy as np

def move_to_str(move):
    colour = move // 21
    direction = (move % 21) // 7
    distance = (move % 7) + 1
    colour_dict = {0: "Brown", 1: "Green", 2: "Red", 3: "Yellow", 4: "Pink", 5: "Purple", 6: "Blue", 7: "Orange"}
    direction_dict = {0: "Left", 1: "Up", 2: "Right"}
    string = colour_dict[colour] + " " + direction_dict[direction] + " " + str(distance)
    return string

board_colours = np.array([[7, 6, 5, 4, 3, 2, 1, 0],
                          [2, 7, 4, 1, 6, 3, 0, 5],
                          [1, 4, 7, 2, 5, 0, 3, 6],
                          [4, 5, 6, 7, 0, 1, 2, 3],
                          [3, 2, 1, 0, 7, 6, 5, 4],
                          [6, 3, 0, 5, 2, 7, 4, 1],
                          [5, 0, 3, 6, 1, 4, 7, 2],
                          [0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int8)

colours_rgb = [
    "#8B4513",  # Brown
    "green",
    "red",
    "yellow",
    "#FF69B4",  # Pink
    "purple",
    "blue",
    "#FFA500"  # Orange
]

circle_black = "●"  # Black Large Circle
circle_fisheye = "◉" # Fisheye Circle

symbols = [circle_black, circle_fisheye]

def _print_board(board):
    console = Console()
    for row in range(8):
        row_lines = ["", "", ""]  # Each square will be 2 rows tall

        for col in range(8):
            bg = colours_rgb[board_colours[row, col]]
            style = Style(bgcolor=bg)
            row_lines[0] += f"[{style}]     [/{style}]"
            row_lines[2] += f"[{style}]     [/{style}]"
            if board[row, col] == -1:
                row_lines[1] += f"[{style}]     [/{style}]"
            else:
                player = board[row, col] // 10
                colour = colours_rgb[board[row, col] % 10]
                symbol = symbols[player]
                if player == 0:
                    checker_style = Style(color=colour, bgcolor="black", bold=True)
                else:
                    checker_style = Style(color=colour, bgcolor="#A9A9A9", bold=True)

                row_lines[1] += f"[{style}]  [/{style}]"
                row_lines[1] += f"[{checker_style}]{symbol}[/{checker_style}]"
                row_lines[1] += f"[{style}]  [/{style}]"

        # Print both rows for square shape
        console.print(row_lines[0])
        console.print(row_lines[1])
        console.print(row_lines[2])


def print_board(board):
    print()
    _print_board(board[:8])
    print()
    colour_to_move = board[8, 0]
    if colour_to_move != -1:
        colour_dict = {0: "Brown", 1: "Green", 2: "Red", 3: "Yellow", 4: "Pink", 5: "Purple", 6: "Blue", 7: "Orange"}
        print(f"Colour to move is {colour_dict[colour_to_move]}")
    return
