import numpy as np
from surfaces.surface import Surface

class InfinitePlane(Surface):
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index
    
    def intersection(self, ray):
        # t = -(P0 • N + d) / (V • N)
        denom = np.dot(self.normal, ray.direction)
        if abs(denom) < 1e-6:
            return None
        t = -(np.dot(self.normal, ray.origin) + self.offset) / denom
        if t < 0:
            return None
        P = ray.origin + t * ray.direction
        return t, P, self.normal
