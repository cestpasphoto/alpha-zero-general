import logging
log = logging.getLogger(__name__)

import bisect
from tqdm import tqdm
from santorini.SantoriniGame import NUMBER_PLAYERS
from santorini.SantoriniDisplay import move_to_str


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False, other_way=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        # if NUMBER_PLAYERS == 2:
        #     players = [self.player2, self.player1]                             if other_way else [self.player1, self.player2]
        # elif NUMBER_PLAYERS == 3:
        #     players = [self.player2, self.player1, self.player1]               if other_way else [self.player1, self.player2, self.player2]
        # elif NUMBER_PLAYERS == 4:
        #     players = [self.player2, self.player1, self.player1, self.player1] if other_way else [self.player1, self.player2, self.player2, self.player2]
        # elif NUMBER_PLAYERS == 5:
        #     players = [self.player2, self.player1, self.player1, self.player1] if other_way else [self.player1, self.player2, self.player2, self.player2]
        players = ([self.player2]+[self.player1]*(NUMBER_PLAYERS-1)) if other_way else ([self.player1]+[self.player2]*(NUMBER_PLAYERS-1))
        curPlayer = 0
        board = self.game.getInitBoard()
        it = 0
        while not self.game.getGameEnded(board).any():
            it += 1
            if verbose:
                if self.display:
                    self.display(board)
                print()
                print(f'Turn {it} Player {curPlayer}: ', end='')
                
            canonical_board = self.game.getCanonicalForm(board, curPlayer)
            action = players[curPlayer](canonical_board)
            valids = self.game.getValidMoves(canonical_board, 0)

            if verbose:
                print(f'P{curPlayer} decided to {move_to_str(action, curPlayer)}')

            if valids[action] == 0:
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            if self.display:
                self.display(board)
            print("Game over: Turn ", str(it), "Result ", self.game.getGameEnded(board))
            
        return self.game.getGameEnded(board)[0]

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        ratio_boundaries = [        1-0.60,        1-0.55,        0.55,        0.60         ]
        colors           = ['#d60000',     '#d66b00',     '#f9f900',   '#a0d600',  '#6b8e00'] #https://icolorpalette.com/ff3b3b_ff9d3b_ffce3b_ffff3b_ceff3b

        oneWon, twoWon, draws = 0, 0, 0
        t = tqdm(range(num), desc="Arena.playGames", ncols=120, disable=None)
        for i in t:
            # Since trees may not be resetted, the first games (1vs2) can't be
            # considered as fair as the last games (2vs1). Switching between 
            # 1vs2 and 2vs1 like below seems more fair:
            # 1 2 2 1   1 2 2 1  ...
            one_vs_two = (i%4 == 0) or (i%4 == 3)
            t.set_description('Arena ' + ('(1 vs 2)' if one_vs_two else '(2 vs 1)'), refresh=False)
            gameResult = self.playGame(verbose=verbose, other_way=not one_vs_two)
            if gameResult == (1. if one_vs_two else -1.):
                oneWon += 1
            elif gameResult == (-1. if one_vs_two else 1.):
                twoWon += 1
            else:
                draws += 1

            t.set_postfix(one_wins=oneWon, two_wins=twoWon, refresh=False)
            ratio = oneWon / (oneWon+twoWon) if oneWon+twoWon>0 else 0.5
            t.colour = colors[bisect.bisect_right(ratio_boundaries, ratio)]
        t.close()

        return oneWon, twoWon, draws
