import numpy as np
from surfaces.surface import Surface
from utils import normalize
class Sphere(Surface):
    def __init__(self, position, radius, material_index):
        self.position = position
        self.radius = radius
        self.material_index = material_index

    def intersection(self, ray):
        sphere_center = np.array(self.position)
        ray_origin = np.array(ray.origin)

        # Vector from ray origin to sphere center
        L = sphere_center - ray_origin

        # Projection of L onto ray direction
        tca = np.dot(L, ray.direction)
        if tca < 0:
            return None
        
        # Distance check from center to ray
        distance_squared = np.dot(L, L) - tca**2
        if distance_squared > self.radius**2:
            return None
        
        # Calculate intersection distances along ray
        thc = np.sqrt(self.radius**2 - distance_squared)
        t0 = tca - thc
        t1 = tca + thc
        if t0 > 0:
            t = t0
        elif t1 > 0:
            t = t1
        else:
            return None
        
        # Hit point coordinates and surface normal
        P = ray_origin + t * ray.direction
        N = normalize(P - sphere_center)
        return t, P, N

