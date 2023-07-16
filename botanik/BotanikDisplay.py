import numpy as np
from colorama import Style, Fore, Back
# from .BotanikLogicNumba import my_unpackbits
from .BotanikConstants import *

#######################
mask = np.array([4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint16)
def my_unpackbits(value):
	return (np.bitwise_and(value.astype(np.uint16), mask) != 0).astype(np.uint8)
#######################

def move_to_str(move, player):
	if move < 15:
		card_i, register_slot = divmod(move, 5)
		return f'take arrival card {card_i}, and put on player register slot {register_slot}'
	elif move < 30:
		card_i, middlerow_slot = divmod(move-15, 5)
		return f'take arrival card {card_i}, and put on middle row slot {middlerow_slot}'
	elif move < 35:
		middlerow_slot = move-30
		return f'swap mecabot with slot {middlerow_slot} of middle row'
	elif move < 37:
		slot = move-35
		return f'use freed card {slot} to expand the machine'
	else:
		return f'unknown move {move}'

############################# PRINT GAME ######################################

#                 EMPTY        SOURCE        BLUE       YELLOW       GREEN      RED        BLACK
print_colors  = [Back.RESET, Back.MAGENTA, Back.BLUE, Back.YELLOW, Back.GREEN, Back.RED, Back.BLACK]
directions_str = [
#  0fl  1fl  3fl
  ['·', '?', '?'], # N=0, E=0, S=0, W=0
  ['╵', '?', '╹'], # N=1, E=0, S=0, W=0
  ['╶', '?', '╺'], # N=0, E=1, S=0, W=0
  ['└', '╚', '?'], # N=1, E=1, S=0, W=0
  ['╷', '?', '╻'], # N=0, E=0, S=1, W=0
  ['│', '║', '?'], # N=1, E=0, S=1, W=0
  ['┌', '╔', '?'], # N=0, E=1, S=1, W=0
  ['├', '╠', '?'], # N=1, E=1, S=1, W=0
  ['╴', '?', '╸'], # N=0, E=0, S=0, W=1
  ['┘', '╝', '?'], # N=1, E=0, S=0, W=1
  ['─', '═', '?'], # N=0, E=1, S=0, W=1
  ['┴', '╩', '?'], # N=1, E=1, S=0, W=1
  ['┐', '╗', '?'], # N=0, E=0, S=1, W=1
  ['┤', '╣', '?'], # N=1, E=0, S=1, W=1
  ['┬', '╦', '?'], # N=0, E=1, S=1, W=1
  ['┼', '╋', '?'], # N=1, E=1, S=1, W=1
]
mecabot_str = '🃟'

statuses_str = [
	'main player to get card from arrival to register',
	'other player to expand his machine',
	'other player to swap his mecabot card',
	'main player to expand his machine',
	'main player to swap a mecabot card',
]

def direction_code(a):
	return min(16, a[0] + 2*a[1] + 4*a[2] + 8*a[3])

def card_to_str(card):
	result = Fore.WHITE + print_colors[card[0]]
	if card[2] == MECABOT:
		result += mecabot_str
	else:
		i_flw = [0, 1, 1, 2][card[1]]
		result += directions_str[direction_code(card[3:])][i_flw]
	result += Fore.RESET + Back.RESET
	return result

def bitfield_to_str(bitfield):
	result = 'Available: '
	# Translate list of available cards to a simple format
	for color in range(5):
		available_cards = my_unpackbits(256*bitfield[0, color].astype(np.uint8) + bitfield[1, color].astype(np.uint8))
		for i, b in enumerate(available_cards):
			if b:
				result += card_to_str(np_all_cards[color, i, :]) + ' '
			else:
				result += '· '
	return result

def _print_main(board):
	print('-'*60)
	print(board.misc[0,:], statuses_str[board.misc[0, 1]], f', main player=P{board.misc[0,2]}')
	print(bitfield_to_str(board.misc[3:,:5]))
	print()
	print('Arrival zone: ', end='')
	for i in range(3):
		print(' ' + card_to_str(board.arrival_cards[i,:]), end='')
	print()

	print('P0:  ', end='')
	for i in range(5):
		print(' ' + card_to_str(board.p0_register[i,:]), end='')
	print()

	print('     ', end='')
	for i in range(5):
		print(' ' + card_to_str(board.middle_reg[i,:]), end='')
	print()

	print('P1:  ', end='')
	for i in range(5):
		print(' ' + card_to_str(board.p1_register[i,:]), end='')
	print()

	print('Freed:', end='')
	for i in range(4):
		p, slot = divmod(i, 2)
		if slot == 0:
			print(f' [P{p}] ', end='')
		print(card_to_str(board.freed_cards[i,:]), end=' ')
	print()
	print('-'*60)

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