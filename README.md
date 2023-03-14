# CPU-optimized Alpha Zero General (any game, any framework!)

Based on the superb repo https://github.com/suragnair/alpha-zero-general but with higher strength, optimized for CPU-only 25-100x speed improvement, and supporting 2+ players games. You can play with it on your browser https://github.com/cestpasphoto/cestpasphoto.github.io.
Games: Splendor, The Little Prince - Make me a planet, Machi Koro (Minivilles), Santorini (with basic gods)

### Splendor

![Sample game of Splendor](splendor/sample_game.gif)

* [x] Support of [Splendor game](https://en.wikipedia.org/wiki/Splendor_(game)) with 2 players
* [x] Support of 3-4 players (just change NUMBER_PLAYERS constant)
* [x] Proper MCTS handling of "chance" factor when revealing new deck card
* [x] Added pretrained models for 2-3-4 players

There are some limitations: implemented logic doesn't allow you to both take gems from the bank and give back some (whereas allowed in real rules), you can either 1-2-3 gems or give back 1-2 gems.

### Machi Koro / Minivilles

* [x] Quick implementation of [Minivilles](https://en.wikipedia.org/wiki/Machi_Koro), with handful limitations

![Sample game of Minivilles with 4 players](minivilles/sample_game.gif)


### The Little Prince - Make me a planet

* [x] Quick implementation of [The little prince](https://cdn.1j1ju.com/medias/67/f8/eb-the-little-prince-make-me-a-planet-rulebook.pdf), with limitations. Main ones are:
   * No support of 2 players, only 3-5 players are supported
   * When market is empty, current player doesn't decide card type, it is randomly chosen.
   * Grey sheeps are displayed on console using grey wolf emoji, and brown sheeps are displayed using a brown goat.


### Santorini

* [x] Own implementation of [Santorini](https://www.ultraboardgames.com/santorini/game-rules.php), policy for initial status is user switchable (predefined, random or chosen by players)
* [x] Support of goddess (basic ones only)

![Sample game of Santorini](santorini/sample_game_with_random_init.gif)

About 70% winrate against [Ai Ai](http://mrraow.com/index.php/aiai-home/aiai/) and 90+% win rate against [BoardSpace AI](https://www.boardspace.net/english/index.shtml). See [more details here](santorini/README.md)

---

## Technical details

<details>
  <summary>Click here for details about training, running or playing</summary>

#### Dependencies

`pip3 install onnxruntime numba tqdm colorama coloredlogs`
and
`pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu`

Contrary to previous investigations, latest versions of onnxruntime and pytorch lead to best performance, see GenericNNetWrapper.py line 315

#### How to play versus saved engine

`./pit.py splendor/pretrained_2players.pt human -n 1`


_Ongoing code/features rework, some pretrained networks won't work anymore_
You can also make 2 networks fight each other ![2 networks fighting](splendor/many_games.gif). Contrary to baseline version, pit.py automatically retrieves training settings and load them (numMCTSSims, num_channels, ...) although you can override if you want; you may even select 2 different architecture to compare them!

#### Recommended settings for training

Compared to initial version, I target a smaller network but more MCTS simulations allowing to see further: this approach is less efficient on GPU, but similar on CPU and allow stronger AI.

`main.py -m 800 -e 500 -i 10 -c 2.5 -F -f 0.03 -T 10 -d 0.50 -b 32 -l 0.0003 -p 2 -C ../results/mytest`: 

* Start by defining proper number of players in SplendorGame.py and disabling card reserve actions in first lines of splendor/SplendorLogicNumba.py
* `-b 32 -l 0.0003 -p 2`: define batch size, learning rate and number of epochs. Larger number of epochs degrades performance, same for larger batch sizes
* `-c 2.5 -f 0.03 -F`: MCTS options, here cpuct value, FPU (first play urgency) and enabled forced playouts

![Sample training](splendor/sample_training.jpg)

Of course you need to tune parameters depending on the game, especially cpuct, FPU, dirichlet noise. The option `-V` allows you to switch between different NN architectures. If you specify a previous checkpoint using a different architecture, it will still try loading weights as much as possible. It allows me starting first steps of training with small/fast networks and then I experiment larger networks.

#### Multithreading

I also usually execute several trainings in parallel; you can evaluate the results obtained in the last 24 hours by using this command (execute as many times as threads): `./pit.py -A 24 -T 8`

The code also runs several games to benefit from faster batch inferences; note that games are not run simultaneously but one at a time, meaning it still uses 1 CPU core. The downside is the bigger memory footprint.
</details>
