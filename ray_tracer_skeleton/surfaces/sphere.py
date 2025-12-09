import numpy as np

class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = position
        self.radius = radius
        self.material_index = material_index

    def intersection(self, ray):
        L = self.position - ray.origin
        tca = np.dot(L, ray.direction)
        if tca < 0:
            return None
        distance_squared = np.dot(L, L) - tca**2
        if distance_squared > self.radius**2:
            return None
        thc = np.sqrt(self.radius**2 - distance_squared)
        t0 = tca - thc
        t1 = tca + thc
        if t0 > 0:
            t = t0
        elif t1 > 0:
            t = t1
        else:
            return None
        P = ray.origin + t * ray.direction
        N = (P - self.position) / np.linalg.norm(P - self.position)
        return t, P, N
