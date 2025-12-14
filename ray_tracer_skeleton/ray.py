import numpy as np

class Ray:
    def __init__(self, pixel_point, camera):
        self.origin = camera.position
        direction = pixel_point - self.origin
        self.direction = direction / np.linalg.norm(direction)