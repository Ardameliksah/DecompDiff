"""Minimal utils package (vendored for DecompDiff / Colab).

The full MyCode utils/__init__ imports training/sampling/visualization modules
that DecompDiff does not use; this stripped version only exposes set_seed so
`MyCode.utils.data_utils` can be imported without those heavy dependencies.
"""
import random, warnings
import numpy as np
import torch


def set_seed(seed, cudnn_deterministic=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
