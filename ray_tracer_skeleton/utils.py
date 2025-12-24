import numpy as np

EPS = 1e-6

def normalize(vector):
    return vector / np.linalg.norm(vector)