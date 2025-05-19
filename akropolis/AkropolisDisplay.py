import re
import numpy as np
from colorama import init, Fore, Style
from .AkropolisConstants import *

# Init Colorama
init(autoreset=True)

# Regex pour retirer les ANSI
_strip = re.compile(r'\x1b\[[0-9;]*m').sub
# Symboles et exposants
SYMS = {DISTRICT: '⌂', PLAZA: '★', QUARRY: 'Q'}
SUP  = {1: '', 2: '²', 3: '³', 4: '⁴'}
# Palette par index (0–7, last color is for QUARRY)
COLS = [Fore.BLUE, Fore.YELLOW, Fore.RED, Fore.MAGENTA, Fore.GREEN, Fore.BLACK]

def strip_ansi(s):
    return _strip('', s)


def prl(board):
    """
    -> List[str] : lignes colorées prêtes à afficher
    board: np.array shape (q_size, r_size, 2) 
           [:,:,0]=type_color, [:,:,1]=height
    """

    lines = []
    for q in range(board.shape[0]):
        tiles = []
        for r in range(board.shape[1]):
            tc, ht = board[q, r]
            t, c = divmod(int(tc), 8)
            if t == 0:
                txt = '·'.center(2)
                col = ''
            else:
                sym = SYMS.get(t, '?') + SUP.get(int(ht), '')
                txt = sym.center(2)
                col = COLS[-1] if t == QUARRY else COLS[c]
            tiles.append(f"{col}{txt}{Style.RESET_ALL}")
        indent = ('  ' * q)
        lines.append(indent + ' '.join(tiles))
    return lines

def print_boards(boards):
    # 1) Prépare les listes de lignes et mesures
    # pss = [prl(b) for b in boards]
    pss = [prl(boards[:,:,0,:]), prl(boards[:,:,1,:])]
    raws = [[strip_ansi(l) for l in ps] for ps in pss]
    widths = [max((len(r) for r in raw), default=0) for raw in raws]
    H = max(len(ps) for ps in pss)
    # 2) Pad en hauteur
    for idx in range(len(pss)):
        pad = H - len(pss[idx])
        pss[idx]  += [''] * pad
        raws[idx] += [''] * pad
    # 3) Affichage
    for i in range(H):
        left = pss[0][i] + ' '*(widths[0] - len(raws[0][i]))
        # if len(boards) == 2:
        right = pss[1][i] + ' '*(widths[1] - len(raws[1][i]))
        print(f"{left}  |  {right}")
        # else:
        #     print(left)

def print_board(board):
    print_boards(board.board)

def move_to_str(move: int, player: int) -> str:
    rem, orient          = divmod(move, N_ORIENTS)
    tile_idx_in_cs, site = divmod(rem, CITY_AREA)
    q, r                 = divmod(site, CITY_SIZE)

    degrees = int(orient * (360 / N_ORIENTS))

    return (
        f"P{player} places tile #{tile_idx_in_cs} "
        f"at position (q={q}, r={r}) "
        f"with orientation {orient} ({degrees}°)."
    )


if __name__ == "__main__":
    # Démo : deux petits boards 5×5 aléatoires
    rng = np.random.RandomState(0)
    boards = []
    for _ in range(2):
        tc = rng.choice([0,1,2], size=(12,12), p=[0.6,0.25,0.15])
        ht = rng.choice([1,2,3], size=(12,12), p=[0.8,0.15,0.05])
        # encode type_color = type*8 + color (ici color=random 0–7)
        clr = rng.randint(0,8,size=(12,12))
        boards.append(np.stack([tc*8 + clr, ht], axis=2))
    print_boards(boards)
