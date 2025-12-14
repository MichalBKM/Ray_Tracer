import numpy as np
from surfaces.surface import Surface

class Cube(Surface):
    def __init__(self, position, scale, material_index):
        self.position = position # Cube center
        self.scale = scale # Edge length
        self.material_index = material_index
    
    # TODO: Implement the intersection method
    def intersection(self, ray):
        origin = ray.origin          # Starting point of the Ray which is a numpy array [ox, oy, oz]
        direction = ray.direction    # Direction vector also a numpy array [dx, dy, dz]

        # 1. Compute cube bounds from center and scale
        half = self.scale / 2.0
        min_bound = self.position - half   # [min_x, min_y, min_z]
        max_bound = self.position + half   # [max_x, max_y, max_z]

        t_min = -np.inf   
        t_max =  np.inf   
        eps = 1e-6

        # 2. Slab method per axis
        for axis in range(3):  # 0=x, 1=y, 2=z
            o = origin[axis]
            d = direction[axis]
            slab_min = min_bound[axis]
            slab_max = max_bound[axis]

            if abs(d) < eps:
                # no Direction on this axis -> Ray is parallel
                if o < slab_min or o > slab_max:
                    return None  # Ray is outside the slab -> no hit
                # else: Move to the next axis 
                continue

            # Where does the ray hit the two planes on this axis
            t1 = (slab_min - o) / d
            t2 = (slab_max - o) / d

            t_near = min(t1, t2)
            t_far  = max(t1, t2)

            # Update global interval by crossing them
            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)

            # no valid interval -> no intersection
            if t_min > t_max:
                return None

        # 3. Check that the intersection is in front of the ray origin
        if t_max < eps:
            return None  # whole box is behind the ray

        # If we start outside the cube, t_min>0 is the entry point.
        # If we start inside, t_min<=0 and t_max is the exit point.
        t_hit = t_min if t_min > eps else t_max

        # 4. Compute hit point: P = origin + direction * t
        P = origin + t_hit * direction

        # 5. Compute normal from which face we hit
        normal = np.zeros(3)
        tol = 1e-4

        if abs(P[0] - min_bound[0]) < tol:
            normal = np.array([-1.0, 0.0, 0.0])
        elif abs(P[0] - max_bound[0]) < tol:
            normal = np.array([1.0, 0.0, 0.0])
        elif abs(P[1] - min_bound[1]) < tol:
            normal = np.array([0.0, -1.0, 0.0])
        elif abs(P[1] - max_bound[1]) < tol:
            normal = np.array([0.0, 1.0, 0.0])
        elif abs(P[2] - min_bound[2]) < tol:
            normal = np.array([0.0, 0.0, -1.0])
        elif abs(P[2] - max_bound[2]) < tol:
            normal = np.array([0.0, 0.0, 1.0])

        return t_hit, P, normal

