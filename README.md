# scalable-dreamcoder
Training scalable (neural) dreamcoder on code contests dataset.

### Method
A scalable (neural) version of Dreamcoder for program synthesis. The discrete latent codebook of a VQ-VAE is used to represent a library of program primitives. Transformer based encoder and decoder are used to encode and reconstruct python programs to and from the codebook. One of the codebook latents is used as a “sink” latent in that no other latents attend to the sink latent during program decoding. Thus driving the encoder embeddings towards the sink latent corresponds to compressing the program encodings (similar to the program abstraction phase in Dreamcoder). A discrete diffusion model (SEDD) is trained to perform search over the codebook latents conditioned on problem statements (similar to training the recognition network in Dreamcoder). Thus given a problem statement, the diffusion model performs discrete program search in the codebook space. The generated latents are then decoded into python programs by the VQ-VAE decoder which are verified by an external compiler (similar to wake phase in Dreamcoder). As new problems are solved by the system during the wake phase, the corresponding generated programs are added to the train dataset. A growing dataset of programs leads to a richer library of primitives during the abstraction phase. Training the diffusion model to search over a richer library of primitives leads to solving further new problems in the next wake phase, thus forming a virtuous continual learning cycle.

### Current Status
Currently the training works but loss goes down super slowly. Possible bottlenecks:
1. Reconstruction loss is over-restrictive and difficult to optimize (can be mitigated by replacing with jepa-like latent prediction)
2. Less available compute (throw more compute at the training)
