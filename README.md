# My Alpha Zero General (any game, any framework!)

Based on the superb repo https://github.com/suragnair/alpha-zero-general, with the following additions:

### Added Features

* [x] Added Dirichlet Noise as per original [DeepMind paper](https://www.nature.com/articles/nature24270.epdf), using this [pull request](https://github.com/suragnair/alpha-zero-general/pull/186)
* [x] Compute policy gradients properly when some actions are invalid based on [A Closer Look at Invalid Action Masking inPolicy Gradient Algorithms](https://arxiv.org/pdf/2006.14171.pdf) and its [repo](https://github.com/vwxyzjn/invalid-action-masking)
* [ ] Improvements based on [Accelerating Self-Play Learning in Go](https://arxiv.org/pdf/1902.10565.pdf)
* [ ] Set up HyperParameters Optimization, like Hyperband or Population-Based Training

### Splendor

* [x] Support of [Splendor game](https://en.wikipedia.org/wiki/Splendor_(game)) with 2 players
* [ ] Support of 3-4 players
* [ ] Proper MCTS handling of "chance" factor when revealing new deck card
* [ ] Optimized implementation of Splendor
* [ ] Explore various architecture

### Others changes

* [ ] Include ELO-like ranking
* [x] Improved prints (logging, tqdm, colored bards depending on current Arena results)
* [x] Parameters can be set in cmdline, added new parameters like time limit

### Contributors and Credits
* [Shantanu Thakoor](https://github.com/ShantanuThakoor) and [Megha Jhunjhunwala](https://github.com/jjw-megha) helped with core design and implementation.
* [Shantanu Kumar](https://github.com/SourKream) contributed TensorFlow and Keras models for Othello.
* [Evgeny Tyurin](https://github.com/evg-tyurin) contributed rules and a trained model for TicTacToe.
* [MBoss](https://github.com/1424667164) contributed rules and a model for GoBang.
* [Jernej Habjan](https://github.com/JernejHabjan) contributed RTS game.
* [Adam Lawson](https://github.com/goshawk22) contributed rules and a trained model for 3D TicTacToe.
* [Carlos Aguayo](https://github.com/carlos-aguayo) contributed rules and a trained model for Dots and Boxes along with a [JavaScript implementation](https://github.com/carlos-aguayo/carlos-aguayo.github.io/tree/master/alphazero).

