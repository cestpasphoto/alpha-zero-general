import re
import numpy as np
from colorama import init, Fore, Style
from .AkropolisConstants import *
from .AkropolisLogicNumba import decode_value_from_int8

init(autoreset=True)
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

def strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences for width calculations."""
    return _ANSI_RE.sub('', s)

# Glyphs for district (house) and plaza (star), plus superscripts for heights
SYMBOLS = {
    EMPTY          : ('·', Fore.BLACK),
    QUARRY         : ('Q', Fore.BLACK),
    PLAZA_BLUE     : ('*', Fore.BLUE),
    PLAZA_YELLOW   : ('⁑', Fore.YELLOW),
    PLAZA_RED      : ('⁑', Fore.RED),
    PLAZA_PURPLE   : ('⁑', Fore.MAGENTA),
    PLAZA_GREEN    : ('⁂', Fore.GREEN),
    DISTRICT_BLUE  : ('⌂', Fore.BLUE),
    DISTRICT_YELLOW: ('⌂', Fore.YELLOW),
    DISTRICT_RED   : ('⌂', Fore.RED),
    DISTRICT_PURPLE: ('⌂', Fore.MAGENTA),
    DISTRICT_GREEN : ('⌂', Fore.GREEN),
}
COLORS = [Fore.BLUE, Fore.YELLOW, Fore.RED, Fore.MAGENTA, Fore.GREEN, Fore.BLACK]
SUPERSCRIPTS = {0: '', 1: '', 2: '²', 3: '³', 4: '⁴', 5: '⁵'}
sub_digits = str.maketrans({
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
    '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
    '-': '₋'
})

def _print_glyph(code, height, center=True):
    glyph, col = SYMBOLS.get(code, ('?', Fore.RED))
    glyph += SUPERSCRIPTS[height]
    tile = glyph.center(4) if center else glyph
    return f"{col}{tile}{Style.RESET_ALL}"

def _print_cs(game):
    hex_to_str = [
        '     ',
        'Quarr',
        'BluDi',
        'YelDi',
        'RedDi',
        'PurDi',
        'GreDi',
        'BluPl',
        'YelPl',
        'RedPl',
        'PurPl',
        'GrePl',
    ]
    result = ''
    for idx in range(CONSTR_SITE_SIZE):
        tile = game.construction_site[idx, :3]
        if tile[0] == EMPTY:
            continue
        if result != '':
            result += '  '
        result += '-'.join([hex_to_str[x] for x in tile])
    return result

def _print_lines(board_descr: np.ndarray, board_height: np.ndarray) -> list[str]:
    lines = []
    for r in range(board_descr.shape[1]):
        row = '  ' if r%2==1 else ''
        for q in range(board_descr.shape[0]):
            code = board_descr[r, q]
            height = int(board_height[r, q])
            if code == EMPTY and q % 3 == 0 and r % 3 == 0:
                # print some coordinates instead of glyph
                coords = f'{r},{q}'.center(4).translate(sub_digits)
                row += f"{Style.DIM}{coords}{Style.RESET_ALL}"
            else:
                row += _print_glyph(code, height)
        lines.append(row)
    return lines

def gen_png(game, filename):
    from .AkroPlot3 import hex_prism_traces
    import plotly.graph_objects as go
    import plotly.io as pio
    from PIL import Image, ImageChops, ImageDraw, ImageFont
    import os
    from io import BytesIO

    color_list = ['white', '#6E6E6E',
        '#4257B2', '#B79526', '#AF4435', '#6B4CA1', '#3C8658',
        '#4257B2', '#B79526', '#AF4435', '#6B4CA1', '#3C8658',
    ]
    n_stars = [0, 0, 
        0, 0, 0, 0, 0,
        1, 2, 2, 2, 3,
    ]

    tmp_files = []
    for p in range(N_PLAYERS):
        fig = go.Figure()
        for r in range(CITY_SIZE):
            for q in range(CITY_SIZE):
                d, h = game.board_descr[r, q, p], game.board_height[r, q, p]
                if d == EMPTY:
                    continue
                h_ = (h)*7
                for t in hex_prism_traces(r=r, q=q, fill_color=color_list[d], H=h_, n_stars=n_stars[d]):
                    fig.add_trace(t)

        # fig.update_layout(
        #     title=dict(
        #         text=f'P{p} - {game.total_scores[p]+SCORE_OFFSET} {game.stones[p]}s {_print_cs(game)}',
        #         x=0.5,
        #         y=0.75,
        #         xanchor="center",
        #         yanchor="top",
        #         font=dict(size=12, color="black")   # style
        #     ),
        #     # margin=dict(t=80)     # marge haute pour que le titre ne chevauche rien
        # )

        fig.update_layout(
            scene=dict(
                aspectmode="data",          # échelle identique X, Y, Z (facultatif)
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            scene_camera=dict(
                eye=dict(x=-2, y=-3, z=5),   # x,y petits → presque au-dessus
                up=dict(x=0.1, y=0.8, z=0.5),          # garde le nord en haut
                center=dict(x=0, y=0, z=0)       # point visé (souvent l’origine)
            ),
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            autosize=False,
            width=600,
            height=600,
        )
        # if int(game.total_scores[0]) + 127 > 10:
        #     fig.show()
        #     breakpoint()
        # pio.write_image(fig, f'tmp_p{p}.png', scale=2)
        png_bytes = fig.to_image(format="png", scale=2)      # besoin de kaleido
        img = Image.open(BytesIO(png_bytes))

        # -- détection bande blanche : tout pixel dont la luminance > 250/255
        arr   = np.array(img)
        lum   = arr.mean(axis=2)                  # luminance naïve
        mask  = lum < 250                         # zones « dessin »
        coords = np.argwhere(mask)

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1           # +1 car slicing ouvert
        img_cropped = img.crop((x0, y0, x1, y1))  # (left, upper, right, lower)
        img_cropped.save(f'tmpcrop2_p{p}.png')

        tmp_files.append(f'tmpcrop2_p{p}.png')

        # breakpoint()

    from math import floor

    CANVAS_W, CANVAS_H  = 1200, 600        # taille finale fixe
    HEADER_H            = 40               # zone réservée au texte dans le canvas
    BG_COLOR            = "white"
    TEXT_COLOR          = "black"
    imgs = [Image.open(f) for f in tmp_files]        # vos PNG intermédiaires

    text = (
        f'P0 {game.stones[0]}st {decode_value_from_int8(game.total_scores[0])}pts'
        f'    -    {_print_cs(game)}    -    '
        f'P1 {game.stones[1]}st {decode_value_from_int8(game.total_scores[1])}pts'
    )
 
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), color=BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    font = ImageFont.load_default(24)
    bbox      = draw.textbbox((0, 0), text, font=font)
    text_w    = bbox[2] - bbox[0]
    text_h    = bbox[3] - bbox[1]
    text_x    = (CANVAS_W - text_w) // 2
    text_y    = (HEADER_H - text_h) // 2
    draw.text((text_x, text_y), text, font=font, fill=TEXT_COLOR)

    # Largeur totale des images
    tot_imgs_w = sum(im.width for im in imgs)
    n          = len(imgs)
    # Espace horizontal = (largeur libre) / (n + 1)
    gap = max(0, floor((CANVAS_W - tot_imgs_w) / (n + 1)))
    x = gap
    for im in imgs:
        # centrage vertical dans la partie « images » (CANVAS_H - HEADER_H)
        y = HEADER_H + (CANVAS_H - HEADER_H - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + gap

    canvas.save(filename)

    # breakpoint()
    for f in tmp_files:
        os.remove(f)

    if game.misc[1] <= 0 and game.construction_site[1, 0] == EMPTY:
        # THE END
        make_video(game, crf=18)

import subprocess
import pathlib
from datetime import datetime
import glob
import os

def make_video(game, crf=18):
    stamp = datetime.now().strftime("%m%d%H%M")
    outfile = f"game_{stamp}_{decode_value_from_int8(game.total_scores[0])}-{decode_value_from_int8(game.total_scores[1])}.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-framerate", "1",
        "-i", "./board_%02d.png",
        # "-vf", '"pad=ceil(iw/2)*2:ceil(ih/2)*2"',
        "-c:v", "libx264",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        outfile
    ]
    subprocess.run(cmd, check=True)

    for f in glob.glob("./board_*.png"):
        try:
            os.remove(f)
        except OSError:
            print(f"⚠ impossible de supprimer {f}")


def print_board(game):
    """
    Print up to N_PLAYERS boards, two side by side per row, with:
      - centered headers "PLAYER i"
      - colored score breakdown P×D per color
      - the hex boards
      - remaining construction site tiles
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
        # generate line lists and raw widths
        lines0 = _print_lines(game.board_descr[:, :, p]  , game.board_height[:, :, p])
        lines1 = _print_lines(game.board_descr[:, :, p+1], game.board_height[:, :, p+1]) if (p+1) < N_PLAYERS else []
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
                terms = [f"{game.plazas[i, c]*PLAZA_STARS[c]}×{game.districts[i, c]}" for c in range(N_COLORS)]
                stones = game.stones[i]
                total = decode_value_from_int8(game.total_scores[i])
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
    remaining = [
        ''.join([_print_glyph(t, 1, center=False) for t in tile[:3]])
        for tile in game.construction_site if tile[0] != EMPTY
    ]
    # remaining = [str(tile) for tile in game.construction_site if tile[0] != EMPTY]
    print("Construction site:", ' '.join(remaining), '  ', int(game.misc[1]), ' stack(s) remaining')

    gen_png(game, f'./board_{game.misc[0]:02}.png')

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
