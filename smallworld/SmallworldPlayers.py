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
                print(action, end=' ')
        print()

    def play(self, board, nb_moves):
        # print_board(self.game.board)
        valids = self.game.getValidMoves(board, 0)
        print()
        print('='*60, 'type your move, or + to get the list of moves')
        while True:
            input_move = input()
            if input_move == '+':
                self.show_all_moves(valids)
            else:
                try:
                    a = int(input_move)
                    if not valids[a]:
                        raise Exception('')
                    break
                except:
                    print('Invalid move:', input_move)
        return a


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
