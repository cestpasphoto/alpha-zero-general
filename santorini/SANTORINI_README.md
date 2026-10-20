# Santorini Rules and Structures

[Santorini](https://en.wikipedia.org/wiki/Santorini_(game)) is a two-player competative board game that takes place on a 5x5 board. Each player has two character pieces, (an infinite number of) cubic building pieces as well as dome pieces. 
Players begin the game by alternatingly placing their character pieces on an empty board space. At this point the board is empty except for the 4 player pieces (2 for each player). 

### Gameplay 
__The gameplay__ involves a player moving one of their pieces one space in any direction (including diagonal) and *then* building a block on a space adjacent to the location they end up at. 

When a block is placed on top of a space, the height of the space is increased by one. 

Multiple blocks can be stacked on top of one another, so if the board itself has a height of zero, a space with two blocks on it has a height of 2. 

A piece can move down any amount of height, but cannot move up by more than 1 in a turn. So a piece cannot move from the board onto the space with two blocks on it, without first moving onto a space next to that one with one block on it, and then waiting for the opponent to perform their turn.

If there are already three blocks on a space, instead of another block, a dome can be built on that space. Pieces cannot stand on, nor build on spaces which have domes. See [this photo](https://www.1843magazine.com/sites/default/files/styles/il_manual_crop_16_9/public/Santorini-header-V3.jpg) for an image of the domed roofs in the actual town of Santorini, Italy.  

### Winning
A player **wins** when they move a piece onto a space of height 3 (having three blocks on it, and no dome), or when their opponent is unable to make a legal move.

### Legal Move requirements
The following must be true about a player's move for it to be legal:
* The player cannot move onto, or build on, any space with another player's piece in it. 
* The player cannot move onto, or build on, any space with a dome on it, regardless of who placed the dome. 
* The block must be built in a space adjacent to where the piece that was moved ended up. The player cannot move one piece and build next to another piece. The player cannot move one piece and build on a space that was adjacent to where the piece started, but not adjacent to where the piece ended.
* The player must move and then must build. If the player either cannot move a piece, or cannot build after moving, that player loses. It is not possible for a player to move a piece onto a winner square, and then be unable to build, because the player will always have the option to build on space they moved from.
* The player cannot move a piece onto a square that is more than 1 block higher than the square the piece moved from.
* The player cannot pass their turn. 
* The player cannot place a build a block on a space that already has three blocks on it. The player can only build a dome on that space, or build elsewhere. 

## Implementation specifics

Board is 5x5 only. Logic is in `SantoriniLogicNumba.py` (numba jitclass), constants and action encoding in `SantoriniConstants.py`.

### State
`int8` array of shape `(5, 5, 3)`, always given from the point of view of the player to move (canonical form):
* `[:,:,0]` workers: `1`, `2` for the player to move, `-1`, `-2` for the opponent, `0` if empty.
* `[:,:,1]` levels: `0`-`3` blocks, `4` = dome.
* `[:,:,2]` gods and metadata, read as a flat array of 25 cells:
  * cells 0-10: player to move, one cell per god id. Value `64` = god owned; lower bits store god-specific data during multi-step turns (Artemis/Demeter: `9*worker + direction + 1`, Prometheus: `9*worker + 1`, Hermes: number of moves already done, Athena: `1` if moved up last turn, Pan: `1` when won by moving down 2+ levels).
  * cells 11-21: same for the opponent.
  * cell 22: round counter (saturates at 127).
  * cell 23: win flag, `+1`/`-1` when player 0/1 moved up onto level 3 (set in terminal states only).

### Gods
`NB_GODS` in `SantoriniConstants.py`: `1` disables gods, `11` enables no-god + Apollo, Minotaur, Atlas, Hephaestus, Artemis, Demeter, Hermes, Pan, Athena, Prometheus. At init each player receives a different random god.
Gods with optional extra steps (Artemis, Demeter, Hermes, Prometheus) split a turn into several sub-turns in which the same player plays again.

### Actions
`action = worker*(NB_GODS*81) + power*81 + move_direction*9 + build_direction`, i.e. `2*NB_GODS*81` actions.
Directions are numbered on a 3x3 grid around the worker (`0 1 2 / 3 4 5 / 6 7 8`), `4` meaning no move / no build.
`power` is the god power used by this action, `NO_GOD` for a plain move + build.

### Initial placement
`INIT_METHOD` in `SantoriniLogicNumba.py`:
* `0`: fixed positions around the center,
* `1` (default): random positions,
* `2`: players place their workers alternately. During this phase, actions `0`-`24` are cell indices (`5*row + col`).

### Winning
A player wins by moving up onto level 3 during their own turn (a worker pushed onto level 3 by Minotaur does not win), with Pan's power, or when the opponent has no legal move.

### Symmetries
8 per position for training: identity, 3 rotations, 2 flips, swap of workers 1/2, swap of workers -1/-2 (6 during the placement phase).
