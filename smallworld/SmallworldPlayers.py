import numpy as np
import random

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, nb_moves):
        valids = self.game.getValidMoves(board, player=0)
        action = random.choices(range(self.game.getActionSize()), weights=valids.astype(np.int8), k=1)[0]
        return action


class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def show_all_moves(self, valids):
        for action, v in enumerate(valids):
            if v:
                print(f'{action} = {self.game.moveToString(action, 0)}')

    def play(self, board, nb_moves):
        valids = self.game.getValidMoves(board, 0)
        print()
        print('=' * 60)
        self.show_all_moves(valids)
        while True:
            input_move = input("your move (number), '+' to list again: ").strip()
            if input_move == '+':
                self.show_all_moves(valids)
                continue
            try:
                a = int(input_move)
            except ValueError:
                print(f"'{input_move}' is not a number.")
                continue
            if not 0 <= a < len(valids) or not valids[a]:
                print(f'{a} is not a legal move right now.')
                continue
            return a


class GreedyPlayer():
    """
    One-ply greedy baseline, a sanity yardstick rather than a serious opponent.

    Evaluation = banked score + points pending on the territories we own
    (territories[:, 6] for areas where territories[:, 7] == 0): get_score()
    alone stays flat during the conquest phase.
    Arena hands over a CANONICAL board, so this player is always player 0.
    A fixed non-zero sim_seed makes each candidate evaluation deterministic.
    Ties are broken at RANDOM so that pit games do not all repeat.
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
