* [x] Support games with **more than 2 players**
* Speed/memory optimized - Reaching **about 3000 rollouts/sec per CPU core**, meaning about 5 sec/game during self-play (using 800 rollouts per move), with an i5 from 2019 without GPU. All in all, that is a 25x to 100x speed improvement compared to initial repo, see [details here](santorini/README.md).
  * [x] MCTS and logic optimized thanks to Numba, NN inference is now >70% time spent during self-plays based on profiler analysis
  * [x] Neural Network inference speed and especially latency improved, thanks to ONNX 
  * [x] Batched MCTS for speed, no use of virtual loss
  * [x] Memory optimized with no performance impact, using zlib compression
* [x] Algorithm improvements based on [Accelerating Self-Play Learning in Go](https://arxiv.org/pdf/1902.10565.pdf)
  * [x] Playout Cap Randomization
* Improve MCTS strength
  * [x] Added Dirichlet Noise as per original [DeepMind paper](https://www.nature.com/articles/nature24270.epdf), using this [pull request](https://github.com/suragnair/alpha-zero-general/pull/186)
  * [x] FPU, based on parent value ([article](https://arxiv.org/pdf/1902.10565.pdf))
  * [x] Learning based Q and Z ([blog](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628))
  * [x] Forced Playout from [KataGo article](https://arxiv.org/pdf/1902.10565.pdf)
  * [x] Compute policy gradients properly when some actions are invalid based on [A Closer Look at Invalid Action Masking inPolicy Gradient Algorithms](https://arxiv.org/pdf/2006.14171.pdf) and its [repo](https://github.com/vwxyzjn/invalid-action-masking)
  * [x] Temperature strategy from [SAI](https://github.com/CuriosAI/sai/issues/8)
  * [x] Optimize number of MCTS move, see [github ticket](https://github.com/leela-zero/leela-zero/issues/1416)
  * [x] MCTS parameters tuning
* Improve NN strength
  * [x] Use blocks from [MobileNetv3](https://arxiv.org/abs/1905.02244) for optimal accuracy with high speed
  * [x] Improve training speed using [OneCycleLR](https://arxiv.org/pdf/1506.01186.pdf) and [AdamW](https://arxiv.org/abs/1711.05101)
  * [x] Upgrade to KL-divergence loss instead of crossentropy

What I tried but didn't worked:
* MCTS: advanced cpuct formula (using init and base), [surprise weight](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md), and handle different training with Z and Q values (not averaging) like [this article](https://doi.org/10.48550/arXiv.2103.17228)
* NN: SGD optimizer, ReduceLROnPlateau scheduler
* NN architecture: Dropout, BatchNorm2D (BatchNorm works), GARB [article](https://www.mdpi.com/2079-9292/10/13/1533), regular architectures like EfficientNet, ResNet, ResNet v2, Squeeze-Excitation, Inception, ResNext, ...
* Performance improvements: new memory allocator (TBB, TC, JE, ...)

Others changes: parameters can be set in cmdline (added new parameters like time limit) and improved prints (logging, tqdm, colored bards depending on current Arena results). Output an ELO-like ranking

Still todo:
  * [ ] PC-PIMC or what I call "universes" (https://doi.org/10.3389/frai.2023.1014561)
  * [ ] Auto dirichlet at each iteration, or auto-dirichlet at each move
  * [ ] Run full random move in 1% of game to increase diversity 
  * [ ] HyperParameters Optimization (like Hyperband or Population-Based Traininginclude)
  * [ ] Multiprocessing to use several cores during self play
  * [ ] KLD-thresholding (https://github.com/LeelaChessZero/lc0/pull/721)
