import random
import numpy as np


def random_augment(damaged, mask, clean):
    if random.random() < 0.5:
        damaged, mask, clean = (np.flip(a, axis=1) for a in (damaged, mask, clean))
    if random.random() < 0.5:
        damaged, mask, clean = (np.flip(a, axis=0) for a in (damaged, mask, clean))
    k = random.choice([0, 1, 2, 3])
    damaged, mask, clean = (np.rot90(a, k) for a in (damaged, mask, clean))
    return damaged, mask, clean