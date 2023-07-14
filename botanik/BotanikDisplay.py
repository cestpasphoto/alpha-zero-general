import numpy as np
from colorama import Style, Fore, Back

def move_to_str(move, player):
	return f'Move'

############################# PRINT GAME ######################################

#                 EMPTY        SOURCE        BLUE       YELLOW       GREEN      RED        BLACK
print_colors  = [Fore.RESET, Fore.MAGENTA, Fore.BLUE, Fore.YELLOW, Fore.GREEN, Fore.RED, Fore.RESET]
#                     0            1                        2                  3
print_flowers = [Back.RESET, Back.LIGHTMAGENTA_EX, Back.LIGHTMAGENTA_EX, Back.MAGENTA]
directions_str = [
  ' ', # N=0, E=0, S=0, W=0
  '╹', # N=1, E=0, S=0, W=0
  '╺', # N=0, E=1, S=0, W=0
  '┗', # N=1, E=1, S=0, W=0
  '╻', # N=0, E=0, S=1, W=0
  '┃', # N=1, E=0, S=1, W=0
  '┏', # N=0, E=1, S=1, W=0
  '┣', # N=1, E=1, S=1, W=0
  '╸', # N=0, E=0, S=0, W=1
  '┛', # N=1, E=0, S=0, W=1
  '╸', # N=0, E=1, S=0, W=1
  '━', # N=1, E=1, S=0, W=1
  '┓', # N=0, E=0, S=1, W=1
  '┨', # N=1, E=0, S=1, W=1
  '┳', # N=0, E=1, S=1, W=1
  '╋', # N=1, E=1, S=1, W=1
]
mecabot_str = '🃟'

def direction_code(a):
	return min(16, a[0] + 2*a[1] + 4*a[2] + 8*a[3])

def card_to_str(card):
	result = print_colors[card[0]]
	if card[1] < 0:
		result += Back.RESET + mecabot_str
	else:
		result += print_flowers[card[1]]
		result += directions_str[direction_code(card[2:])]
	result += Fore.RESET + Back.RESET
	return result

def _print_main(board):
	print(board.misc[0,:])
	print('P0:  ', end='')
	for i in range(5):
		print(' ' + card_to_str(board.p0_cards[i,:]), end='')
	print()

	print('     ', end='')
	for i in range(5):
		print(' ' + card_to_str(board.middle_row[i,:]), end='')
	print()

	print('P1:  ', end='')
	for i in range(5):
		print(' ' + card_to_str(board.p1_cards[i,:]), end='')
	print()

def print_board(board):
	print()
	# _print_round_and_scores(board)
	_print_main(board)

# def random_card():
# 	color = np.random.randint(0,7)
# 	nbflw = np.random.choice([-1, 0, 1, 3])
# 	direc = np.random.randint(0,2, size=4)
# 	card = np.array([color, nbflw, direc[0], direc[1], direc[2], direc[3]])
# 	return card

# for _ in range(10):
# 	c = random_card()
# 	print(c)
# 	print(card_to_str(c))
# 	print()