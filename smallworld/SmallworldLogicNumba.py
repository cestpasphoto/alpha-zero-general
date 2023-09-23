import numpy as np
from numba import njit
import numba

from SmallworldConstants import *
from SmallworldDisplay import print_board, print_valids

############################## BOARD DESCRIPTION ##############################


# state = [

#nbPPL TypePPL  Total  Player
#  [0,   NOPPL  , 0   , -1 ], # 0
#  [0,   NOPPL  , 1   , -1 ], # 1
#  [2,   -PRIMIT, 2   , -1 ], # 2
#  [0,   NOPPL  , 0   , -1 ], # 3
#  [2,   -PRIMIT, 2   , -1 ], # 4

#nbPPL TypePPL   status      ?
#  [7, OGRE   , TO_CONQUEST, 0 ], # Active ppl in player 0's hand
#  [9, DWARF  , TO_CONQUEST, 0 ], # Active ppl in player 1's hand

#  ?    TypePPL    ?      ?
#  [0,   -HUMAN , 0  ,   0 ], # Declined ppl in player 0's hand
#  [0,   NOPPL  , 0  ,   0 ], # Declined ppl in player 1's hand

# score  turn    ?      ?
#  [10,   0    , 0  ,   0 ], # Score of player 0
#  [9 ,   0    , 0  ,   0 ], # Score of player 1

# nbPPL TypePPL   cost   ?
# [ 5,   HUMAN,   1  ,   0 ]
# [ 5,   TYPE2,   2  ,   0 ]
# [ 5,   TYPE3,   3  ,   0 ]
# [ 5,   TYPE4,   4  ,   0 ]
# [ 5,   TYPE5,   5  ,   0 ]
# [ 5,   TYPE6,   6  ,   0 ]
# [ bitfieldPPL, bitfieldPPL, 0, 0 ] 14 ppl (2octets) + 20 pouvoirs (3octets)

# ]


# @njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (NB_AREAS + 3*NUMBER_PLAYERS + DECK_SIZE+1, 4)

# @njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return 10


mask = np.array([32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint16)

# @njit(cache=True, fastmath=True, nogil=True)
def my_packbits(array):
	product = np.multiply(array.astype(np.uint16), mask[:len(array)])
	return product.sum()

# @njit(cache=True, fastmath=True, nogil=True)
def my_unpackbits(value):
	return (np.bitwise_and(value, mask) != 0).astype(np.uint8)

# @njit(cache=True, fastmath=True, nogil=True)
def my_random_choice(prob):
	result = np.searchsorted(np.cumsum(prob), np.random.random(), side="right")
	return result


spec = [
	('state'         , numba.int8[:,:]),
	('territories'   , numba.int8[:,:]),
	('active_ppl'    , numba.int8[:,:]),
	('declined_ppl'  , numba.int8[:,:]),
	('scores'    	 , numba.int8[:,:]),
	('people_deck'	 , numba.int8[:,:]),
]
# @numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.state = np.zeros(observation_size(), dtype=np.int8)
		self.init_game()

	def get_score(self, player):
		return self.scores[player, 0]

	def init_game(self):
		self.copy_state(np.zeros(observation_size(), dtype=np.int8), copy_or_not=False)
		for i in range(len(descr)):
			if descr[i][3]:
				nb_primit = initial_nb_people[PRIMIT]
				self.territories[i,:] = [nb_primit, PRIMIT, nb_primit, -1]
			else:
				self.territories[i,:] = [0, NOPPL, 0, -1]
			if descr[i][0] == MOUNTAIN:
				self.territories[i,2] += 1

		# Init deck of people and draw for P0 and P1
		self._init_deck()
		chosen_ppl = self.people_deck[0, :]
		self.active_ppl[0,:]   = [chosen_ppl[0], chosen_ppl[1], NEW_TURN_STARTED , 0]
		self._update_deck_after_chose(0)
		chosen_ppl = self.people_deck[0, :]
		self.active_ppl[1,:]   = [chosen_ppl[0], chosen_ppl[1], WAITING_OTHER_PL , 0]
		self._update_deck_after_chose(0)

		self.declined_ppl[0,:] = [0, NOPPL , 0                , 0]
		self.declined_ppl[1,:] = [0, NOPPL , 0                , 0]

		# First round is round #1
		self._update_round()
	
	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.territories  = self.state[0                        :NB_AREAS                 ,:]
		self.active_ppl   = self.state[NB_AREAS                 :NB_AREAS+  NUMBER_PLAYERS,:]
		self.declined_ppl = self.state[NB_AREAS+  NUMBER_PLAYERS:NB_AREAS+2*NUMBER_PLAYERS,:]
		self.scores       = self.state[NB_AREAS+2*NUMBER_PLAYERS:NB_AREAS+3*NUMBER_PLAYERS,:]
		self.people_deck  = self.state[NB_AREAS+3*NUMBER_PLAYERS:NB_AREAS+3*NUMBER_PLAYERS+DECK_SIZE+1,:]

	def valid_moves(self, player):
		result = np.zeros(10, dtype=np.bool_)
		return result

	def make_move(self, move, player, deterministic):
		return 1-player

	def get_state(self):
		return self.state

	def get_round(self):
		return self.scores[0,1]

	def check_end_game(self, next_player):
		if self.get_round() <= 10:
			return np.array([0, 0], dtype=np.float32) # No winner yet

		# Game is ended
		scores = self.scores[:, 0]
		best_score = scores.max()
		return np.where(scores == best_score, 1, -1).astype(np.float32)

	def swap_players(self, nb_swaps):
		return

	def get_symmetries(self, policy, valids):
		# Always called on canonical board, meaning player = 0
		# In all symmetries, no need to update the "optim" vectors as they are not used by NN
		symmetries = [(self.state.copy(), policy.copy(), valids.copy())]
		state_backup, policy_backup, valids_backup = symmetries[0]

		return symmetries

	###########################################################################

	def _valids_attack(self, player):
		valids = np.zeros(NB_AREAS, dtype=np.bool_)

		# Check that player has a valid active ppl:
		if self.active_ppl[player, 1] == NOPPL:
			return valids

		# Attack permitted when it's the time
		if self.active_ppl[player, 2] not in [NEW_TURN_STARTED, JUST_ATTACKED, JUST_ABANDONED]:
			return valids

		# Check that player has at least 1 player in hand, and have a chance to win
		territories_of_player = self._are_owned_by_player(player, active_ppl_only=True)
		nb_ppl_of_player = self.active_ppl[player,0]
		if self.active_ppl[player, 2] == JUST_ATTACKED:
			# Forbid to continue if 0 ppl left in hand
			if nb_ppl_of_player <= 0:
				return valids
		else:
			# Simulate redeploy: add people on the boards, except 1 per territory
			nb_ppl_to_get_from_board = np.dot(np.maximum(self.territories[:,0]-1,0), territories_of_player)
			nb_ppl_of_player += nb_ppl_to_get_from_board

		for area in range(NB_AREAS):
			valids[area] = self._valid_attack_area(player, area, nb_ppl_of_player, territories_of_player)

		return valids

	def _valid_attack_area(self, player, area, nb_ppl_of_player, territories_of_player):
		# No attack on water
		if descr[area][0] == WATER:
			return False

		# No attack on its own people (allow to self attach on declined territory)
		if self._is_owned_by_player(area, player, active_ppl_only=True):
			return False

		# Check that player has at least 1 player in hand, and have a chance to win
		minimum_ppl_for_attack = self.territories[area,2] + 2
		if nb_ppl_of_player + MAX_DICE < minimum_ppl_for_attack:
			return False

		# Check that territory is close to another owned territory or is on the edge
		nb_territories_of_player = np.count_nonzero(territories_of_player)
		if nb_territories_of_player == 0:
			area_is_on_edge = descr[area][2]
			if not area_is_on_edge:
				return False
		else:
			neighbor_areas = connexity_matrix[area]
			if not np.any(np.logical_and(neighbor_areas, territories_of_player)):
				return False

		return True

	def _do_attack(self, player, area):
		# Prepare people if 1st action of the turn
		if self.active_ppl[player, 2] != JUST_ATTACKED:
			self._gather_active_ppl_but_one(player)

		nb_ppl_of_player, type_ppl_of_player = self.active_ppl[player,0], self.active_ppl[player,1]
		minimum_ppl_for_attack = self.territories[area,2] + 2

		# Use dice if people are needed
		use_dice = (nb_ppl_of_player < minimum_ppl_for_attack)
		if use_dice:
			dice = DICE_VALUES[3]
			if nb_ppl_of_player + dice < minimum_ppl_for_attack:
				print(f'  Using dice, random value is {dice} but fails')
				# self._gather_active_ppl_but_one(player)
				self.active_ppl[player, 2] = TO_START_REDEPLOY
				return
			print(f'  Using dice, random value is {dice} and succeed')

		# Attack is successful

		# Update loser and winner
		self._give_back_to_loser(area)
		nb_attacking_ppl = min(minimum_ppl_for_attack, nb_ppl_of_player)
		self._player_wins_territory(area, player, nb_attacking_ppl)

		# Update winner's status
		self.active_ppl[player, 2] = JUST_ATTACKED
		# Redeploy if that was last attack
		if self.active_ppl[player, 0] == 0 or use_dice:
			self._switch_from_attack_to_deploy(player)

	def _valids_redeploy(self, player):
		valids = np.zeros(NB_AREAS + MAX_REDEPLOY, dtype=np.bool_)

		# Check that it is time
		if self.active_ppl[player, 2] not in [TO_START_REDEPLOY, TO_REDEPLOY]:
			if self.active_ppl[player, 2] in [JUST_DECLINED, WAITING_OTHER_PL]:
				return valids

		# Check there is at least one active territory
		territories = self._are_owned_by_player(player, active_ppl_only=True)
		nb_territories = np.count_nonzero(territories)
		if nb_territories == 0:
			# If no other option, then allow to skip redeploy
			valids[0] = True
			return valids

		# Check that player has still some ppl to deploy
		if self.active_ppl[player, 2] == TO_REDEPLOY:
			how_many_ppl_available = self.active_ppl[player, 0]
		else:
			how_many_ppl_available = self._ppl_virtually_available(player, territories)
		if how_many_ppl_available <= 0:
			# If no other option, then allow to skip redeploy
			valids[0] = True
			return valids

		for ppl_to_deploy in range(1, MAX_REDEPLOY):
			valids[ppl_to_deploy] = self._valid_redeploy_on_each(player, ppl_to_deploy, how_many_ppl_available, nb_territories)
		for area in range(NB_AREAS):
			valids[MAX_REDEPLOY + area] = self._valid_redeploy_area(player, area, territories)

		if not valids.any():
			# If no other option, then allow to skip redeploy
			valids[0] = True

		return valids

	def _valid_redeploy_area(self, player, area, territories):
		# Check that this territory is owned
		if not territories[area]:
			return False
		return True

	def _valid_redeploy_on_each(self, player, how_many_to_deploy, how_many_ppl_available, nb_territories_of_player):
		if how_many_ppl_available < how_many_to_deploy*nb_territories_of_player:
			return False
		return True

	def _do_redeploy(self, player, param):
		if param == 0:
			# Special case, skip redeploy
			self._score_and_switch_to_next_player(player)
			return

		if self.active_ppl[player, 2] != TO_REDEPLOY:
			self._gather_active_ppl_but_one(player)
			self.active_ppl[player, 2] = TO_REDEPLOY

		if param < MAX_REDEPLOY:
			# Deploy X ppl on all active areas
			how_many_to_deploy = param

			territories_of_player = self._are_owned_by_player(player, active_ppl_only=True)
			nb_territories_of_player = np.count_nonzero(territories_of_player)
			self.active_ppl[player, 0] -= how_many_to_deploy*nb_territories_of_player
			assert(self.active_ppl[player, 0] >= 0)
			self.territories[:, 0] += how_many_to_deploy*territories_of_player
			self.territories[:, 2] += how_many_to_deploy*territories_of_player
		else:
			# Deploy 1 ppl on 1 area
			area = param - MAX_REDEPLOY

			self.active_ppl[player, 0]   -= 1
			self.territories[area, 0]    += 1
			self.territories[area, 2]    += 1

		# Trigger end of turn if no more to redeploy
		if self.active_ppl[player, 0] == 0:
			self._score_and_switch_to_next_player(player)

	def _valid_decline(self, player):
		# Going to decline permitted only on 1st move
		if self.active_ppl[player, 2] != NEW_TURN_STARTED:
			return False
		if self.active_ppl[player, 1] == NOPPL:
			return False
		return True

	def _do_decline(self, player):
		# Remove previous declined ppl from the board
		for area in range(NB_AREAS):
			if self._is_owned_by_player(area, player, active_ppl_only=False) and self.territories[area,1] < 0:
				self.territories[area] = [0, NOPPL, 0, -1]

		# Move ppl to decline and keep only 1 ppl per territory
		self._gather_active_ppl_but_one(player)
		self.declined_ppl[player, :] = self.active_ppl[player, :]
		self.declined_ppl[player, 0] = 0
		self.declined_ppl[player, 2] = 0
		
		# Flip back ppl tokens on the board
		for area in range(NB_AREAS):
			if self.territories[area, 1] == self.declined_ppl[player, 1]:
				self.territories[area, 1] = -self.declined_ppl[player, 1]
		self.declined_ppl[player, 1] = -self.declined_ppl[player, 1]

		self.active_ppl[player, :] = [0, NOPPL, WAITING_OTHER_PL, 0]
		self._score_and_switch_to_next_player(player)

	def _valids_choose_ppl(self, player):
		valids = np.zeros(DECK_SIZE, dtype=np.bool_)

		# Check that it is time
		if self.active_ppl[player, 2] != NEW_TURN_STARTED:
			return valids
		# Check that player hasn't a player yet
		if self.active_ppl[player, 1] != NOPPL:
			return valids

		for index in range(DECK_SIZE):
			# Check that index is valid
		 	valids[index] = (self.people_deck[index, 1] != NOPPL)

		return valids

	def _do_choose_ppl(self, player, index):
		chosen_ppl = self.people_deck[index, :]
		self.active_ppl[player,:] = [chosen_ppl[0], chosen_ppl[1], NEW_TURN_STARTED , 0]
		self._update_deck_after_chose(index)

	def _valids_abandon(self, player):
		valids = np.zeros(NB_AREAS, dtype=np.bool_)

		# To avoid infinite loops, allow to abandon area only at the beginning of the turn
		if self.active_ppl[player, 2] != NEW_TURN_STARTED:
			return valids
		if self.active_ppl[player, 1] == NOPPL:
			return valids

		for area in range(NB_AREAS):
			valids[area] = self._valid_abandon_area(player, area)

		return valids

	def _valid_abandon_area(self, player, area):
		if not self._is_owned_by_player(area, player, active_ppl_only=False):
			return False
		return True

	def _do_abandon(self, player, area):
		self._leave_area(area)
		self.active_ppl[player, 2] = JUST_ABANDONED

	###########################################################################

	def _is_owned_by_player(self, area, player, active_ppl_only=False):
		if active_ppl_only:
			result = (self.territories[area,3] == player) and (self.territories[area,1] == self.active_ppl[player, 1])
		else:
			result = (self.territories[area,3] == player)
		return result

	def _are_owned_by_player(self, player, active_ppl_only=False):
		if active_ppl_only:
			result = np.logical_and(self.territories[:,3] == player, self.territories[:,1] == self.active_ppl[player, 1])
		else:
			result = (self.territories[:,3] == player)
		return result

	def _give_back_to_loser(self, area):
		loser = self.territories[area,3]
		if loser >= 0:
			if self._is_owned_by_player(area, loser, active_ppl_only=True):
				self.active_ppl[loser, 0] += max(self.territories[area,0]-1, 0)
			else:
				# do not give back is loser ppl is declined
				assert(self.territories[area,1] == self.declined_ppl[loser, 1])

	def _leave_area(self, area):
		# Give back ppl to owner
		owner = self.territories[area,3]
		if self._is_owned_by_player(area, owner, active_ppl_only=True):
			self.active_ppl[owner, 0] += self.territories[area,0]
		else:
			self.declined_ppl[owner, 0] += self.territories[area,0]

		# Make the area empty
		self.territories[area,:] = [0, NOPPL, 0, -1]

	def _player_wins_territory(self, area, player, nb_attacking_ppl):
		self.territories[area,0] = nb_attacking_ppl
		self.territories[area,1] = self.active_ppl[player,1]
		self.territories[area,2] = self.territories[area,0] + (1 if descr[area][0] == MOUNTAIN else 0)
		self.territories[area,3] = player

		self.active_ppl[player, 0] -= nb_attacking_ppl
		assert(self.active_ppl[player, 0] >= 0)	

	def _gather_active_ppl_but_one(self, player):
		# Gather all active people in player's hand, leaving only 1 on each territory
		print(f'Prepare / redeploy P{player}:', end='')
		for area in range(NB_AREAS):
			if self._is_owned_by_player(area, player, active_ppl_only=True):
				nb_ppl_to_gather = max(self.territories[area,0] - 1, 0)
				if nb_ppl_to_gather > 0:
					self.territories[area,0]    -= nb_ppl_to_gather
					self.territories[area,2]    -= nb_ppl_to_gather
					self.active_ppl[player, 0]  += nb_ppl_to_gather
					print(f' {nb_ppl_to_gather}ppl on area{area}', end='')
		print('')

	def _ppl_virtually_available(self, player, player_territories=None):
		if player_territories is None:
			player_territories = self._are_owned_by_player(player, active_ppl_only=True)

		how_many_ppl_available = self.active_ppl[player, 0]
		# Simulate redeploy: add people on the boards, except 1 per territory
		how_many_ppl_available += np.dot(np.maximum(self.territories[:,0]-1,0), player_territories)
		return how_many_ppl_available

	def _switch_from_attack_to_deploy(self, player):
		# self._gather_active_ppl_but_one(player)
		self.active_ppl[player, 2] = TO_START_REDEPLOY

	def _score_and_switch_to_next_player(self, player, new_status_of_cur_player=WAITING_OTHER_PL):
		self._update_score(player)
		self.active_ppl[player, 2] = new_status_of_cur_player

		next_player = 1 - player
		if next_player == 0:
			self._update_round()
		if self.active_ppl[next_player, 2] == WAITING_OTHER_PL:
			self.active_ppl[next_player, 2] = NEW_TURN_STARTED
		else:
			print(f'Wrong status for {next_player}: {self.active_ppl[next_player, 2]}')
			raise Exception

	def _update_score(self, player):
		owned_areas = self._are_owned_by_player(player, active_ppl_only=False)
		score_for_this_turn = 0
		for area in [area for area in range(NB_AREAS) if owned_areas[area]]:
			score_for_this_turn += self.territories[area, 0]
		# print(f'  P{player} scored {score_for_this_turn} at this turn')
		self.scores[player][0] += score_for_this_turn

	def _update_round(self):
		self.scores[:,1] += 1

	def _init_deck(self):
		# All people available except NOPPL and PRIMIT
		available_people = np.ones(PRIMIT, dtype=np.int8)
		available_people[NOPPL] = False

		# Draw 6 ppl randomly
		for i in range(DECK_SIZE):
			chosen_ppl = my_random_choice(available_people / available_people.sum())
			self.people_deck[i, :] = [initial_nb_people[chosen_ppl], chosen_ppl, 0, 0]
			available_people[chosen_ppl] = False

		# Update bitfield
		self.people_deck[DECK_SIZE, :2] = divmod(my_packbits(available_people), 256)

	def _update_deck_after_chose(self, index):
		# Read bitfield
		available_people = my_unpackbits(256*self.people_deck[DECK_SIZE, 0] + self.people_deck[DECK_SIZE, 1])

		# Delete people #item and shift others upwards
		self.people_deck[index:DECK_SIZE-1, :] = self.people_deck[index+1:DECK_SIZE, :]
		# Draw a new people for last combination
		chosen_ppl = my_random_choice(available_people / available_people.sum())
		self.people_deck[DECK_SIZE-1, :] = [6, chosen_ppl, 0, 0]
		available_people[chosen_ppl] = False

		# Update back the bitfield
		self.people_deck[DECK_SIZE, :2] = divmod(my_packbits(available_people), 256)

###############################################################################
import random

b = Board(NUMBER_PLAYERS)
print_board(b)
print()

def play_one_turn():
	p = (b.active_ppl[:, 2] < WAITING_OTHER_PL).nonzero()[0].item()
	print(f'Player is now P{p}')

	while b.active_ppl[p, 2] < WAITING_OTHER_PL:
		valids_attack    = b._valids_attack(player=p)
		valids_abandon   = b._valids_abandon(player=p)
		valids_redeploy  = b._valids_redeploy(player=p)
		valids_choose    = b._valids_choose_ppl(player=p)
		valid_decline    = b._valid_decline(player=p)
		print_valids(p, valids_attack, valids_abandon, valids_redeploy, valids_choose, valid_decline)

		if any(valids_attack) or any(valids_abandon) or valid_decline:
			values = np.concatenate((valids_attack.nonzero()[0], valids_abandon.nonzero()[0], ([2*NB_AREAS] if valid_decline else [])), axis=None)
			dice = np.random.choice(values.astype(np.int64))
			if dice < NB_AREAS:
				area = dice
				print(f'Attacking area {area}')
				b._do_attack(player=p, area=area)
			elif dice < 2*NB_AREAS:
				area = dice - NB_AREAS
				print(f'Abandonning area {area}')
				b._do_abandon(player=p, area=area)
			else:
				print(f'*** Decline current ppl')
				b._do_decline(player=p)

		elif any(valids_choose):
			chosen_ppl = np.random.choice(valids_choose.nonzero()[0].astype(np.int64))
			print(f'Choose ppl #{chosen_ppl}')
			b._do_choose_ppl(player=p, index=chosen_ppl)	

		elif any(valids_redeploy):
			valids_on_each = valids_redeploy[:MAX_REDEPLOY]

			# Chose a "redeploy on each" action if possible
			if valids_on_each.any():
				param = valids_on_each.nonzero()[0].max()
				print(f'Redeploy {param}ppl on each area')
			else:
				area = np.random.choice(valids_redeploy[MAX_REDEPLOY:].nonzero()[0].astype(np.int64))
				param = area + MAX_REDEPLOY
				print(f'Redeploy on area {area}')
			
			b._do_redeploy(player=p, param=param)

		else:
			breakpoint()
			break 

		print('-'*40)
		print_board(b)
		print('-'*40)

	return 1-p

p = 0
while not b.check_end_game(p).any():
	p = play_one_turn()

print(f'The end: {b.check_end_game(p)}')