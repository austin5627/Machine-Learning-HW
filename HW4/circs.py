import math
import numpy as np


def circs():
    x = np.zeros((2, 100))
    for y in range(50):
        i = y * math.pi / 25
        x[0, y] = math.cos(i)
        x[0, y + 50] = math.cos(i)
        x[1, y] = math.sin(i)
        x[1, y + 50] = math.sin(i)
    return x
