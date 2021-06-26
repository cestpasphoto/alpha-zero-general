import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

from splendor.SplendorGame import SplendorGame as Game
from splendor.SplendorGame import NUMBER_PLAYERS
from splendor.SplendorLogic import print_board, move_to_str
from splendor.SplendorLogicNumba import Board, observation_size
from splendor.SplendorPlayers import RandomPlayer
from splendor.NNet import NNetWrapper as NNet
from utils import *
from MCTS import MCTS

import numpy as np
import base64
import json

best_NN = {
	2: '/home/best/dev/results/it11_T20/best2.pt',
	3: '/home/best/dev/results/3A_it8_v2_m1600_bis/best.pt',
	3: '/home/best/dev/results/4D_it4_m1600_T20_v5b/best.pt',
}[NUMBER_PLAYERS]

def deserialize(state_str):
	try:
		state_bytes = base64.b64decode(state_str.replace('-','+').replace('_','/'))
		state_int = [x for x in state_bytes]
		state_np = np.array(state_int, dtype=np.int8).reshape(observation_size(NUMBER_PLAYERS))
		
		game = Game()
		game.board.copy_state(state_np, True)
		return game
	except:
		print('erreur: ' + state_str)
		exit(219)


def serialize(game):
	state_bytes = game.board.state.tobytes();
	state_str = base64.b64encode(state_bytes).decode('ascii').replace('+','-').replace('/','_')
	return state_str;

def next_best_move(game, nn_name, player):
	nn_args = dict(lr=None, dropout=0., epochs=None, batch_size=None, nn_version=-1, save_optim_state=False)
	net = NNet(game, nn_args)
	cpt_dir, cpt_file = os.path.split(nn_name)
	additional_keys = net.load_checkpoint(cpt_dir, cpt_file)
	mcts_args = dotdict({
		'numMCTSSims'     : additional_keys.get('numMCTSSims', 800),
		'cpuct'           : additional_keys.get('cpuct'      , 1.0),
		'prob_fullMCTS'   : 1.,
		'forced_playouts' : False,
	})
	mcts = MCTS(game, net, mcts_args)
	
	backup_state = game.board.state.copy()
	canonical_board = game.getCanonicalForm(game.board.state, player=player)

	# Find best move
	result = np.argmax(mcts.getActionProb(canonical_board, temp=0, force_full_search=True)[0])
	result = int(result)

	game.board.copy_state(backup_state, True)
	game.getNextState(game.board.state, player=player, action=result)
	return result

def next_random_move(game, player):
	backup_state = game.board.state.copy()
	# Pick random move
	random_player = RandomPlayer(game)
	result = random_player.play(game.board.state, player=player)

	game.board.copy_state(backup_state, True)
	game.getNextState(game.board.state, player=player, action=result)
	return result	

def main():
	import argparse
	parser = argparse.ArgumentParser(description='tester')  

	parser.add_argument('--state-base64', '-s' , action='store', default='', help='Status in base64 encoding')
	parser.add_argument('--player'      , '-p' , action='store', default=0 , help='Index of player to play', type=int)
	parser.add_argument('--move'        , '-m' , action='store', default=-1 , help='Play with the specified move (do nothing if not possible)', type=int)
	parser.add_argument('--slow'        , '-S' , action='store_true', help='Use real thinking insteam of random choice to decide next move')
	parser.add_argument('--display'     , '-d' , action='store_true', help='Just display')
	parser.add_argument('--test'               , action='store', default=0 , help='Enable test mode', type=int)
	args = parser.parse_args()

	if args.test > 0:
		print(json.dumps({'best_action': 19, 'new_state': 'aaa'}))
	elif args.display:
		game = deserialize(args.state_base64)
		print_board(game.board)
	elif args.move >= 0:
		game = deserialize(args.state_base64)
		game.getNextState(game.board.state, player=args.player, action=args.move)
		new_state = serialize(game)
		print(json.dumps({
			'best_action': args.move, 
			'new_state': new_state, 
			'prev_player': args.player,
			'action_str': move_to_str(args.move)
		}))
	else:
		game = deserialize(args.state_base64)
		# print_board(game.board)
		
		# if args.slow:
		if True:
			best_move = next_best_move(game, best_NN, player=args.player)
		else:
			best_move = next_random_move(game, player=args.player)			
		# print(f'best action is {best_move} = {move_to_str(best_move)}')
		# print_board(game.board)

		new_state = serialize(game)
		# print(new_state)
		print(json.dumps({
			'best_action': best_move, 
			'new_state': new_state, 
			'prev_player': args.player,
			'action_str': move_to_str(best_move)
		}))

if __name__ == "__main__":
	main()
