import re
import numpy as np
from colorama import init, Fore, Style
from .AkropolisConstants import *

init(autoreset=True)
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

def strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences for width calculations."""
    return _ANSI_RE.sub('', s)

# Glyphs for district (house) and plaza (star), plus superscripts for heights
SYMBOLS      = {QUARRY: 'Q', PLAZA: '★', DISTRICT: '⌂'}
SUPERSCRIPTS = {1: '', 2: '²', 3: '³', 4: '⁴'}
sub_digits = str.maketrans({
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
    '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
    '-': '₋'
})


# Palette
COLORS = [
    Fore.BLUE, Fore.YELLOW, Fore.RED, Fore.MAGENTA, Fore.GREEN,
    Fore.BLACK # for Quarry
]

def _print_glyph(type_, color, height, center=True):
    if type_ == EMPTY:
        glyph = '·'
        col   = ''
    else:
        glyph = SYMBOLS.get(type_, '?') + SUPERSCRIPTS.get(height, '')
        col   = COLORS[-1] if type_ == QUARRY else COLORS[color]
    tile = glyph.center(4) if center else glyph
    return f"{col}{tile}{Style.RESET_ALL}"

def _print_lines(board: np.ndarray) -> list[str]:
    lines = []
    for r in range(board.shape[1]):
        row = '  ' if r%2==1 else ''
        for q in range(board.shape[0]):
            type_, color = divmod(board[r, q, 0], 8)
            height = int(board[r, q, 1])
            if type_ == EMPTY and q % 3 == 0 and r % 3 == 0:
                # print some coordinates instead of glyph
                coords = f'{r},{q}'.center(4).translate(sub_digits)
                row += f"{Style.DIM}{coords}{Style.RESET_ALL}"
            else:
                row += _print_glyph(type_, color, height)
        lines.append(row)
    return lines

def print_board(game):
    """
    Print up to N_PLAYERS boards, two side by side per row, with:
      - centered headers "PLAYER i"
      - colored score breakdown P×D per color
      - the hex boards
      - remaining construction site tiles
    Assumes:
      game.board.shape         == (size, size, N_PLAYERS, 2)
      game.plazas.shape        == (N_PLAYERS, N_COLORS)
      game.districts.shape     == (N_PLAYERS, N_COLORS)
      game.construction_site   == (CONSTR_SITE_SIZE,)
    """
    def center_cols(texts: list[str], widths: list[int]) -> str:
        """Center each text in its visible-width column, join with separators."""
        cols = []
        for txt, w in zip(texts, widths):
            raw = strip_ansi(txt)
            pad = w - len(raw)
            left = pad // 2
            right = pad - left
            cols.append(' ' * left + txt + ' ' * right)
        line = cols[0]
        for col in cols[1:]:
            line += '  |  ' + col
        return line

    for p in range(0, N_PLAYERS, 2):
        # extract per-player sub-boards (size, size, 2)
        sub0 = game.board[:, :, p, :]
        sub1 = game.board[:, :, p+1, :] if (p+1) < N_PLAYERS else None

        # generate line lists and raw widths
        lines0 = _print_lines(sub0)
        lines1 = _print_lines(sub1) if sub1 is not None else []
        raw0   = [strip_ansi(l) for l in lines0]
        raw1   = [strip_ansi(l) for l in lines1]
        w0     = max((len(r) for r in raw0), default=0)
        w1     = max((len(r) for r in raw1), default=0)
        height = max(len(lines0), len(lines1))

        # headers
        headers = [f"PLAYER {i}" if i < N_PLAYERS else '' for i in (p, p+1)]
        print(center_cols(headers, [w0, w1]))

        # score breakdown
        color_scores = []
        for i in (p, p+1):
            if i >= N_PLAYERS:
                color_scores.append('')
            else:
                terms = [f"{game.plazas[i, c]}×{game.districts[i, c]}" for c in range(N_COLORS)]
                stones = game.stones[i]
                total = game.total_scores[i]
                colored = ' + '.join(f"{COLORS[c]}{terms[c]}{Style.RESET_ALL}" for c in range(N_COLORS)) + f" + {stones} = {total}"
                color_scores.append(colored)
        print(center_cols(color_scores, [w0, w1]))

        # board rows
        for row_idx in range(height):
            part0 = lines0[row_idx] if row_idx < len(lines0) else ''
            part1 = lines1[row_idx] if row_idx < len(lines1) else ''
            pad0 = w0 - len(strip_ansi(part0))
            pad1 = w1 - len(strip_ansi(part1))
            print(f"{part0}{' ' * pad0}  |  {part1}{' ' * pad1}")
        print()

    # construction site
    # remaining = [str(t) for t in game.construction_site if t != -1]
    remaining = [
        ''.join([_print_glyph(t//8, t%8, 1, center=False) for t in TILES_DATA[tile_id, :3]])
        for tile_id in game.construction_site if tile_id != -1
        ]
    print("Construction site:", ' '.join(remaining))


def move_to_str(move: int, player: int) -> str:
    tile_idx_in_cs, rem  = divmod(move, N_PATTERNS)
    idx, orient          = divmod(rem, N_ORIENTS)
    r, q = divmod(idx, CITY_SIZE)

    degrees = int(orient * (360 / N_ORIENTS))

    return (
        f"place tile #{tile_idx_in_cs} "
        f"at position (r={r}, q={q}) "
        f"with orientation {orient} ({degrees}°)."
    )
