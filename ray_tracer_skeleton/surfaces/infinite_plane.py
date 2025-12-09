import numpy as np

class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index
    
    def intersection(self, ray):
        # t = -(P0 • N + d) / (V • N)
        try:
            t = -(np.dot(self.normal, ray.origin) + self.offset) / (np.dot(self.normal, ray.direction))
        except ZeroDivisionError:
            return None
        p = ray.origin + t * ray.direction
        return p
