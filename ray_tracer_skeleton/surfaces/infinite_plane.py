import numpy as np
from surfaces.surface import Surface
from utils import EPS

class InfinitePlane(Surface):
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index
    
    def intersection(self, ray):
        # t = -(P0 • N + d) / (V • N)
        norm = np.array(self.normal)
        origin = np.array(ray.origin)
        direction = np.array(ray.direction)

        denom = np.dot(norm, direction)
        if abs(denom) < EPS:
            return None
    
        t = (-np.dot(norm, origin) + self.offset) / denom
        if t < EPS:
            return None
        
        P = origin + t * direction
        return t, P, norm
