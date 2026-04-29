# Data

CIFAR-10 is downloaded automatically by `torchvision.datasets.CIFAR10` the
first time `code/training/dataset.py` is invoked. The download lands here
under `data/cifar-10-batches-py/`.

If you prefer manual download:

```
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xzf cifar-10-python.tar.gz
```

## Channel-wise statistics

The Cold Diffusion paper fits a 1-component GMM on the channel-wise mean
of the dataset for unconditional generation. For CIFAR-10 the channel
means are roughly `(0.491, 0.482, 0.447)` with std `(0.247, 0.244, 0.262)`
in `[0,1]` space — these are computed on the fly in
`code/generation/cold_gen.py`, no manual step required.
