import logging
log = logging.getLogger(__name__)

import bisect
from tqdm import trange
import zlib
import base64
from os import environ

from MCTS import MCTS

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: FACTORIES, i.e. zero-arg callables returning a player
                        function (board, move number) -> action. Called once
                        for perfect-info games, once per seat for hidden-info
                        games, so that seats never share one MCTS tree.
            game: Game object
            display: a function that takes board as input and prints it.
                     Necessary for verbose mode.

        See pit.py for pitting human players/other baselines with each other.
        """
        self.game = game
        self.display = display
        self.macos_terminal = (environ.get("TERM_PROGRAM", "") == "Apple_Terminal" and "ITERM_SESSION_ID" not in environ)
        n_extra = max(game.getNumberOfPlayers() - 1, 1)
        per_seat = hasattr(game, 'getObservation')
        self._p1_pool = [player1() for _ in range(n_extra if per_seat else 1)]
        self._p2_pool = [player2() for _ in range(n_extra if per_seat else 1)]

    def playGame(self, initial_state="", verbose=False, other_way=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        # player1 takes seat 0 and player2 all others, or the reverse when other_way
        n_other = self.game.getNumberOfPlayers() - 1
        if not other_way:
            players = [self._p1_pool[0]] + [self._p2_pool[i % len(self._p2_pool)] for i in range(n_other)]
        else:
            players = [self._p2_pool[0]] + [self._p1_pool[i % len(self._p1_pool)] for i in range(n_other)]
        curPlayer, it = 0, 0
        board = self.game.getInitBoard()
        opening = []  # first plies, for duplicate-game detection

        # Load initial state
        if initial_state != "":
            from numpy import frombuffer, int8
            data = zlib.decompress(base64.b64decode(initial_state), wbits=-15)
            board = frombuffer(data[:-3], dtype=int8).reshape(board.shape)
            curPlayer, it = int(data[-3]), int.from_bytes(data[-2:])

        while not self.game.getGameEnded(board, curPlayer).any():
            it += 1
            if verbose:
                if self.display:
                    self.display(board)
                print()
                print(f'Turn {it} Player {curPlayer}: ', end='')        
                
            canonical_board = self.game.getCanonicalForm(board, curPlayer)
            action = players[curPlayer](canonical_board, it)
            if len(opening) < 10:
                opening.append(int(action))
            valids = self.game.getValidMoves(canonical_board, 0)

            if verbose:
                print(f'P{curPlayer} decided to {self.game.moveToString(action, curPlayer)}')

            if valids[action] == 0:
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action, random_seed=0)
            curPlayer = int(curPlayer)

        if verbose:
            if self.display:
                self.display(board)
            print("Game over: Turn ", str(it), "Result ", self.game.getGameEnded(board, curPlayer))
        else:
            if initial_state != "":
                print(f"Game over: {self.game.getScore(board, 0)} - {self.game.getScore(board, 1)}")

        MCTS.reset_all_search_trees()
            
        return self.game.getGameEnded(board, curPlayer)[0], tuple(opening)

    def playGames(self, num, initial_state="", verbose=False):
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
        if self.macos_terminal:
            colors = ['RED', 'MAGENTA', 'YELLOW', 'CYAN', 'GREEN']

        oneWon, twoWon, draws = 0, 0, 0
        openings = set()
        t = trange(num, desc="Arena.playGames", ncols=120, disable=None)
        for i in t:
            # Seats alternate as 1 2 2 1  1 2 2 1 ... so that neither side always
            # plays with the freshest tree
            one_vs_two = (i%4 == 0) or (i%4 == 3) or (initial_state != "")
            t.set_description('Arena ' + ('(1 vs 2)' if one_vs_two else '(2 vs 1)'), refresh=False)
            gameResult, opening = self.playGame(verbose=verbose, initial_state=initial_state, other_way=not one_vs_two)
            openings.add(opening)
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

        # Opening uniqueness: low values mean near-duplicate games, so an effective
        # sample size far below `num` and an unreliable pit.
        played = oneWon + twoWon + draws
        if played:
            uniq = len(openings) / played
            msg = f"Opening uniqueness: {uniq:.0%} ({len(openings)}/{played} distinct first-10-ply lines)"
            if uniq < 0.20:
                msg += "  <-- WARNING: effective N collapsed, pit likely unreliable"
            print(msg)

        return oneWon, twoWon, draws
