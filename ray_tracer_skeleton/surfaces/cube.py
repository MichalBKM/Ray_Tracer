import numpy as np
from surfaces.surface import Surface
from utils import EPS

class Cube(Surface):
    def __init__(self, position, scale, material_index):
        self.position = position 
        self.scale = scale 
        self.material_index = material_index
    
    def intersection(self, ray):
        origin = ray.origin          
        direction = ray.direction 

        # 1. Define axis boundaries
        half = self.scale / 2.0
        pos_array = np.array(self.position)
        min_bound = pos_array - half  
        max_bound = pos_array + half  

        t_min = -np.inf   
        t_max =  np.inf   

        # 2. Slab method per axis
        for axis in range(3):  
            o = origin[axis]
            d = direction[axis]
            slab_min = min_bound[axis]
            slab_max = max_bound[axis]

            if abs(d) < EPS:
                # no Direction on this axis -> Ray is parallel
                if o < slab_min or o > slab_max:
                    return None  # Ray is outside the slab -> no hit
                # else: Move to the next axis 
                continue

            # Calculate distances to near and far planes
            t1 = (slab_min - o) / d
            t2 = (slab_max - o) / d

            t_near = min(t1, t2)
            t_far  = max(t1, t2)

            
            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)

            # no valid interval means no Intersection
            if t_min > t_max:
                return None

        # 3. Ignore hits behind the Ray
        if t_max < EPS:
            return None 

        # If we start outside the cube, t_min>0 is the entry point.
        # If we start inside, t_min<=0 and t_max is the exit point.
        t_hit = t_min if t_min > EPS else t_max

        # 4. Compute hit point: P = origin + direction * t
        P = origin + t_hit * direction

        # 5. Compute normal from which face we hit
        normal = np.zeros(3)

        if abs(P[0] - min_bound[0]) < EPS:
            normal = np.array([-1.0, 0.0, 0.0])
        elif abs(P[0] - max_bound[0]) < EPS:
            normal = np.array([1.0, 0.0, 0.0])
        elif abs(P[1] - min_bound[1]) < EPS:
            normal = np.array([0.0, -1.0, 0.0])
        elif abs(P[1] - max_bound[1]) < EPS:
            normal = np.array([0.0, 1.0, 0.0])
        elif abs(P[2] - min_bound[2]) < EPS:
            normal = np.array([0.0, 0.0, -1.0])
        elif abs(P[2] - max_bound[2]) < EPS:
            normal = np.array([0.0, 0.0, 1.0])

        return t_hit, P, normal

