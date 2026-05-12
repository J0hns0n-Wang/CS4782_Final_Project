# Data

CIFAR-10 is downloaded automatically by `torchvision.datasets.CIFAR10` the
first time `src/training/dataset.py` is invoked. The download lands here
under `data/cifar-10-batches-py/`.

If you prefer manual download:

```
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xzf cifar-10-python.tar.gz
```
