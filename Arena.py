import logging
log = logging.getLogger(__name__)

import bisect
from tqdm import trange
import zlib
import base64
from os import environ

import numpy as np

from MCTS import MCTS


def report_opening_uniqueness(n_distinct, n_played, what):
    """
    Share of distinct openings, printed only below 90%: near-duplicate games
    shrink the effective sample size far below the number of games played.
    An opening is the sequence of (state, action) of the first plies, so that
    chance outcomes count: same actions with different dice are different games.
    """
    if n_played == 0:
        return
    uniq = n_distinct / n_played
    if uniq >= 0.90:
        return
    msg = f"{what}: opening uniqueness {uniq:.0%} ({n_distinct}/{n_played} distinct openings)"
    if uniq < 0.20:
        msg += "  <-- WARNING: effective N collapsed"
    print(msg)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None, openings=None):
        """
        Input:
            player 1,2: FACTORIES, i.e. zero-arg callables returning a player
                        function (board, move number) -> action. Called once
                        for perfect-info games, once per seat for hidden-info
                        games, so that seats never share one MCTS tree.
            game: Game object
            display: a function that takes board as input and prints it.
                     Necessary for verbose mode.
            openings: None, or (n_plies, sampler1, sampler2). Each opening is drawn by a sampler
                      (canonical board, valid moves) -> action over n_plies plies, then played twice with
                      the seats swapped: sampler1 draws the openings of games 4k and 4k+1, sampler2 those of
                      games 4k+2 and 4k+3. Diversifies deterministic games without adding randomness to
                      the evaluated moves.

        See pit.py for pitting human players/other baselines with each other.
        """
        self.game = game
        self.display = display
        self.macos_terminal = (environ.get("TERM_PROGRAM", "") == "Apple_Terminal" and "ITERM_SESSION_ID" not in environ)
        n_extra = max(game.getNumberOfPlayers() - 1, 1)
        per_seat = hasattr(game, 'getObservation')
        self._p1_pool = [player1() for _ in range(n_extra if per_seat else 1)]
        self._p2_pool = [player2() for _ in range(n_extra if per_seat else 1)]
        self.openings = openings

    def draw_opening(self, sampler, n_plies):
        """(board, curPlayer, plies played, opening records) after n_plies plies drawn by sampler.
        Drawn again if the game ends during the opening."""
        while True:
            board, curPlayer, records = np.copy(self.game.getInitBoard()), 0, []
            for _ in range(n_plies):
                canonical_board = self.game.getCanonicalForm(board, curPlayer)
                valids = self.game.getValidMoves(canonical_board, 0)
                action = sampler(canonical_board, valids)
                assert valids[action] > 0, f'illegal opening action {action}'
                records.append((zlib.crc32(board.tobytes()), int(action)))
                board, curPlayer = self.game.getNextState(board, curPlayer, action, random_seed=0)
                board, curPlayer = np.copy(board), int(curPlayer)   # getNextState may return an internal buffer
                if self.game.getGameEnded(board, curPlayer).any():
                    break
            else:
                return board, curPlayer, n_plies, records

    def playGame(self, initial_state="", verbose=False, other_way=False, start=None):
        """
        Executes one episode of a game, from the initial board, from initial_state, or from start
        (a drawn opening: board, current player, plies already played, opening records).

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
        if start is not None:
            board, curPlayer, it, opening = np.copy(start[0]), start[1], start[2], list(start[3])

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
                opening.append((zlib.crc32(board.tobytes()), int(action)))
            valids = self.game.getValidMoves(canonical_board, 0)

            if verbose:
                print(f'P{curPlayer} decided to {self.game.moveToString(action, curPlayer)}')

            assert valids[action] > 0, f'illegal action {action}'
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
        n_plies, start = (self.openings[0] if self.openings else 0), None
        drawn = []
        t = trange(num, desc="Arena.playGames", ncols=120, disable=None)
        for i in t:
            # Seats alternate as 1 2 2 1  1 2 2 1 ... so that neither side always
            # plays with the freshest tree
            one_vs_two = (i%4 == 0) or (i%4 == 3) or (initial_state != "")
            t.set_description('Arena ' + ('(1 vs 2)' if one_vs_two else '(2 vs 1)'), refresh=False)
            if n_plies and i % 2 == 0:   # a new opening, played by this game and the next one, seats swapped
                start = self.draw_opening(self.openings[1 + (i // 2) % 2], n_plies)
                drawn.append(tuple(start[3]))
            gameResult, opening = self.playGame(verbose=verbose, initial_state=initial_state, other_way=not one_vs_two,
                                                start=start)
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

        report_opening_uniqueness(len(openings), oneWon + twoWon + draws, 'Arena')
        if drawn:   # each drawn opening is played twice: games of a pair may stay identical, so also count the openings
            print(f"Arena: {len(set(drawn))}/{len(drawn)} distinct drawn openings ({n_plies} plies, each played twice)")

        return oneWon, twoWon, draws
