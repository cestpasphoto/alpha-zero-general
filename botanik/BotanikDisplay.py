import numpy as np
from colorama import Style, Fore, Back
from .BotanikConstants import *

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
	elif move < 36+8*MACHINE_SIZE*MACHINE_SIZE:
		card_i, move_ = divmod(move-35, 4*MACHINE_SIZE*MACHINE_SIZE)
		slot, orient = divmod(move_, 4)
		return f'expand machine on slot {slot} with {"first" if card_i == 0 else "second"} free card {["", "rotated 90°", "rotated 180°", "rotated -90°"][orient]}'
	else:
		return f'Unknown move {move}'

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

def machine_to_str(machine):
	result = ''
	for y in range(5):
		for x in range(5):
			result += card_to_str(machine[y,x,:])
		result += '\n'
	return result

def machines_to_str(machine0, machine1):
	result = ''
	for y in range(MACHINE_SIZE):
		for x in range(MACHINE_SIZE):
			result += card_to_str(machine0[y,x,:])
		result += ' '*10
		for x in range(MACHINE_SIZE):
			result += card_to_str(machine1[y,x,:])
		result += '\n'
	return result

def _print_main(board):
	# print('-'*60)
	print('  ', board.misc[0,:], statuses_str[board.misc[0, 1]], f', main player=P{board.misc[0,2]}')
	print('Scores:', board.misc[1, :2])
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

	print('P0 machine:    P1 machine:')
	print(machines_to_str(board.p0_machine, board.p1_machine))

	# print('-'*60)

def print_board(board):
	# print()
	_print_main(board)

# Used for debug purposes
def print_valids(valids, freed_cards_of_player):
	result = ''
	cardinals = ['1st', '2nd', '3rd', '4th']
	orient_str = ['0°', '90°', '180°', '-90°']

	if np.any(valids[:30]):
		# List arrival card to be moved to player register
		for card_i in range(3):
			if np.any(valids[card_i*5:card_i*5+5]):
				result += f'{cardinals[card_i]} arrival on'
				for i in range(5):
					if valids[card_i*5+i]:
						result += f' R{i}'
				result += '. '
		# List where cards can NOT be put on middle register
		result += 'All cards on whole middle row'
		for card_i in range(3):
			if not np.any(valids[15+card_i*5:15+card_i*5+5]):
				result += f' except {cardinals[card_i]} '
			elif not np.all(valids[15+card_i*5:15+card_i*5+5]):
				result += f' except {cardinals[card_i]} on'
				for i in range(5):
					if not valids[15+card_i*5+i]:
						result += f' M{i}'
		result += '.'
	elif np.any(valids[30:35]):
		result = 'Swap mecabot with '
		for i in range(5):
			if valids[30+i]:
				result += f' M{i}'
		result += '.'
	elif np.any(valids[35:-1]):
		mm = MACHINE_SIZE*MACHINE_SIZE
		for card_i in range(2):
			if np.any(valids[35+4*mm*card_i:35+4*mm*card_i+4*mm]):
				result += f'Expand using {cardinals[card_i]} card on slots Y,X'
				for slot in range(mm):
					index = 35 + 4*mm*card_i + 4*slot
					if np.any(valids[index:index+4]):
						result += f' {divmod(slot, MACHINE_SIZE)} ('
						if np.all(valids[index:index+4]):
							result += 'all'
						else:
							for orient in range(4):
								if valids[index+orient]:
									card = freed_cards_of_player[card_i, :].copy()
									card[NORTH:] = np.roll(card[NORTH:], orient)
									result += card_to_str(card) + ' '
						result += ')'
		if valids[-1]:
				result += f'Throw away 1st card, cant expand machine'
	else:
		result = 'No action possible'
	
	result += '\n' + '-' * 80
	print(result)
