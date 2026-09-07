import numpy as np
import random

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, nb_moves):
        valids = self.game.getValidMoves(board, player=0)
        action = random.choices(range(self.game.getActionSize()), weights=valids.astype(np.int8), k=1)[0]
        return action


# ---------------------------------------------------------------------------
# Move logging (diagnostic only)
#
# Every move is appended to HUMAN_LOG_PATH as an independent pickle record
#     (board, action, tag, timestamp, move_id)
# Reading back = repeated pickle.load() until EOFError. Appending record by
# record means a crash or a Ctrl-C mid-game still leaves every earlier move
# on disk.
#
# The board handed to play() is the CANONICAL board (see GreedyPlayer's
# docstring), i.e. exactly what the network is fed, so an offline script can
# replay each position through any checkpoint without further bookkeeping.
# It must be copied: the Board object views that array in place and later
# calls would mutate what we stored.
#
# move_id is "g<game>.r<round>.m<n>", shown on screen while you play so you
# can note a position and inspect it later with
#     analyze_human_games.py --show g4.r8.m3
# The numbering rule is the same one the offline script applies, and the
# logger REPLAYS the existing file on startup, so ids stay consistent across
# sessions appending to the same log. Known limit: a new game is detected by
# the mover's round going back down, so two logs concatenated by hand, or two
# processes writing at once, can confuse the game counter.
#
# Set SMALLWORLD_LOG=<path> to change the destination, or to '' to disable.
# ---------------------------------------------------------------------------
import os
import pickle
import time

HUMAN_LOG_PATH = os.environ.get('SMALLWORLD_LOG', 'human_games.log')


class _MoveLogger:
    """Assigns g<game>.r<round>.m<n> ids and appends records."""

    def __init__(self):
        self.ready = False
        self.game_no, self.prev_round, self.n_in_round = 1, None, 0

    @staticmethod
    def _mover_round(board):
        """Round of the player to move, read from game_status[0,3].
        Row layout mirrors Board.copy_state(): territories, peoples (3*P),
        deck, round_status (P), game_status (P)."""
        from .SmallworldConstants import NUMBER_PLAYERS, DECK_SIZE
        from .SmallworldMaps import NB_AREAS
        return int(np.asarray(board)[NB_AREAS + 4 * NUMBER_PLAYERS + DECK_SIZE, 3])

    def _advance(self, board):
        rnd = self._mover_round(board)
        if self.prev_round is not None:
            if rnd < self.prev_round:        # round went back down: new game
                self.game_no += 1
                self.n_in_round = 0
            elif rnd > self.prev_round:      # new round of the same game
                self.n_in_round = 0
        self.n_in_round += 1
        self.prev_round = rnd
        return f'g{self.game_no}.r{rnd}.m{self.n_in_round}'

    def _catch_up(self):
        """Replay an existing log once, so a resumed session keeps numbering."""
        self.ready = True
        if not HUMAN_LOG_PATH or not os.path.exists(HUMAN_LOG_PATH):
            return
        try:
            with open(HUMAN_LOG_PATH, 'rb') as f:
                while True:
                    try:
                        record = pickle.load(f)
                    except EOFError:
                        break
                    self._advance(record[0])     # record = (board, action, tag, ts, id)
        except Exception as e:
            print(f'[log] could not replay {HUMAN_LOG_PATH}, ids restart at g1: {e}')

    def next_id(self, board):
        """Id of the move about to be played. Call EXACTLY once per move."""
        if not self.ready:
            self._catch_up()
        try:
            return self._advance(board)
        except Exception:
            return '?'

    def write(self, board, action, tag, move_id):
        if not HUMAN_LOG_PATH:
            return
        try:
            with open(HUMAN_LOG_PATH, 'ab') as f:
                pickle.dump((np.asarray(board).copy(), int(action), tag,
                             time.time(), move_id), f)
        except Exception as e:      # logging must never break a game
            print(f'[log] could not write {HUMAN_LOG_PATH}: {e}')


_LOGGER = _MoveLogger()


def next_move_id(board):
    return _LOGGER.next_id(board)


def log_move(board, action, tag, move_id=None):
    """move_id: pass the value returned by next_move_id() if you already asked
    for it (otherwise the counter would advance twice for the same move)."""
    if move_id is None:
        move_id = _LOGGER.next_id(board)
    _LOGGER.write(board, action, tag, move_id)
    return move_id


class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def show_all_moves(self, board, valids):
        """Grouped, annotated move list. Falls back to bare indices if the
        rich description fails, so a display bug can never block a game."""
        try:
            from .SmallworldDisplay import describe_moves
            for line in describe_moves(board, valids):
                print(line)
            return
        except Exception as e:
            print(f'(detailed list unavailable: {e})')
        for action, v in enumerate(valids):
            if v:
                print(f'{action} = {self.game.moveToString(action, 0)}')

    def play(self, board, nb_moves):
        valids = self.game.getValidMoves(board, 0)
        move_id = next_move_id(board)        # once per move, before any redisplay
        print()
        print('=' * 70, move_id)
        self.show_all_moves(board, valids)
        while True:
            input_move = input(f"[{move_id}] your move (number), 'l' to list again, "
                               "'b' for the board: ").strip()
            if input_move in ('+', 'l'):
                self.show_all_moves(board, valids)
                continue
            if input_move == 'b':
                try:
                    from .SmallworldDisplay import state_to_str
                    print(state_to_str(board))
                except Exception as e:
                    print(f'(board unavailable: {e})')
                continue
            try:
                a = int(input_move)
            except ValueError:
                print(f"'{input_move}' is not a number.")
                continue
            if not 0 <= a < len(valids):
                print(f'{a} is out of range (0-{len(valids)-1}).')
                continue
            if not valids[a]:
                print(f'{a} = {self.game.moveToString(a, 0)} -- not legal right now.')
                continue
            break
        log_move(board, a, 'human', move_id)
        return a


class LoggingPlayer():
    """
    Optional: wraps ANY player so its moves are logged too.
    Use it in pit.py if you can wrap the NN player there in one line:
        player2 = LoggingPlayer(player2, 'agent')
    Not required -- logging your own moves alone already answers the main
    question ("what prior does the agent give to the moves I choose?").
    """
    def __init__(self, inner, tag='agent'):
        self.inner, self.tag = inner, tag

    def play(self, board, nb_moves):
        # Snapshot BEFORE playing: some players (GreedyPlayer) run simulations
        # that write into the array we were handed.
        snapshot = np.asarray(board).copy()
        a = self.inner.play(board, nb_moves)
        log_move(snapshot, a, self.tag)
        return a

    def __getattr__(self, name):     # forward anything else to the wrapped player
        return getattr(self.inner, name)


class GreedyPlayer():
    """
    One-ply greedy baseline, meant as a sanity yardstick ("does the champion
    crush a trivial opponent?"), not as a serious opponent.

    Evaluation
    ----------
    board.get_score() alone is useless here: it reads game_status[player, 6],
    a counter only updated when a turn is scored, so it stays flat during the
    conquest phase and every candidate action would look identical.

    Instead we use the board's own per-territory pending points:
        territories[:, 6] = points of the territory if its owner scored NOW
        territories[:, 7] = owning player (-1 if none)
    so the evaluation is
        already banked score  +  points pending on territories we own
    which reacts to every single conquest, abandon or redeploy.

    Conventions
    -----------
    Arena hands over a CANONICAL board, so this player is always player 0 and
    simulation mirrors what MCTS.search does internally: getNextState(board, 0,
    action). getNextState copies the state before mutating it, so the board we
    were given is never corrupted by our simulations.

    A fixed non-zero sim_seed makes each candidate evaluation deterministic
    (random_seed=0 means real randomness in this codebase). Arena still resolves
    the move actually played with its own seed, so this stays an estimate.

    Ties are broken at RANDOM on purpose: a deterministic greedy would make
    every pit game identical and collapse the opening-uniqueness check.
    """

    WIN_BONUS = 10000.

    def __init__(self, game, sim_seed=1):
        self.game = game
        # action_size = 5*NB_AREAS + MAX_REDEPLOY(8) + DECK_SIZE(6) + decline(1) + end(1)
        self.nb_areas = (game.getActionSize() - 16) // 5
        self.sim_seed = sim_seed

    def _value(self, board):
        ended = self.game.getGameEnded(board, 0)
        if ended.any():                       # +1 win / -1 loss / 0.01 shared win
            return self.WIN_BONUS * float(ended[0])
        territories = board[:self.nb_areas, :]
        mine = territories[:, 7] == 0
        pending = float(territories[mine, 6].sum())
        return float(self.game.getScore(board, 0)) + pending

    def play(self, board, nb_moves):
        candidates = np.flatnonzero(self.game.getValidMoves(board, 0))
        if candidates.size == 0:
            raise Exception('GreedyPlayer: no legal move in a non-terminal state')
        if candidates.size == 1:
            return int(candidates[0])

        values = np.empty(candidates.size, dtype=np.float64)
        for i, a in enumerate(candidates):
            next_board, _ = self.game.getNextState(board, 0, int(a), random_seed=self.sim_seed)
            values[i] = self._value(next_board)

        best = candidates[np.flatnonzero(values == values.max())]
        return int(random.choice(list(best)))
