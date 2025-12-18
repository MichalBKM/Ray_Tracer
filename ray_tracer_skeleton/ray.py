import numpy as np
from utils import normalize

class Ray:
    def __init__(self, pixel_point, camera):
        self.origin = camera.position
        direction = pixel_point - self.origin
        self.direction = normalize(direction)