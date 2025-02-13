# scalable-dreamcoder
Training scalable (neural) dreamcoder on code contests dataset.
Currently the training works but loss goes down super slowly. Possible bottlenecks:
1. Reconstruction loss is over-restrictive and difficult to optimize (can be mitigated by replacing with jepa-like latent prediction)
2. Less available compute (throw more compute at the training)
