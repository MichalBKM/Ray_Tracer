import numpy as np
from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere 

def almost_equal(a, b, tol=1e-5):
    return np.allclose(a, b, atol=tol)

def test_plane_intersection():
    # Plane: y = 1 -> n = (0,1,0), offset = -1  (i.e. n·x + d = 0 => y - 1 = 0)
    plane = InfinitePlane(np.array([0.0, 1.0, 0.0]), -1.0, 0)

    # 1) Ray from below, pointing up: should hit at (0,1,0), t = 1
    r1 = Ray(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    hit1 = plane.intersection(r1)
    print("Plane Test 1:", hit1)
    assert hit1 is not None, "Test 1: expected a hit"
    t1, P1, N1 = hit1
    assert abs(t1 - 1.0) < 1e-5, "Test 1: t should be 1"
    assert almost_equal(P1, np.array([0.0, 1.0, 0.0])), "Test 1: P should be (0,1,0)"
    assert almost_equal(N1, np.array([0.0, 1.0, 0.0])), "Test 1: N should be (0,1,0)"

    # 2) Ray parallel above the plane: no hit
    r2 = Ray(np.array([0.0, 2.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    hit2 = plane.intersection(r2)
    print("Plane Test 2:", hit2)
    assert hit2 is None, "Test 2: should be None (parallel and above)"

    # 3) Ray starting above, pointing up (plane is below): intersection has t < 0 -> no visible hit
    r3 = Ray(np.array([0.0, 2.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    hit3 = plane.intersection(r3)
    print("Plane Test 3:", hit3)
    assert hit3 is None, "Test 3: should be None (plane behind ray direction)"

    # 4) Ray starting exactly on the plane, pointing up: t should be 0 or ignored (depending on your epsilon)
    r4 = Ray(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    hit4 = plane.intersection(r4)
    print("Plane Test 4:", hit4)
    # Here, depending on your implementation, either:
    # - you treat t=0 as "no hit" (return None), or
    # - you return t≈0 and P≈(0,1,0).
    # So we just check it doesn't crash.

    print("All plane intersection tests ran.")

def test_cube_intersection():
    # Axis-aligned cube centered at origin, edge length 2
    # So bounds are: x,y,z ∈ [-1, 1]
    cube = Cube(np.array([0.0, 0.0, 0.0]), 2.0, 0)

    # 1) Ray from z = -3 straight towards center → hit front face at (0,0,-1)
    r1 = Ray(np.array([0.0, 0.0, -3.0]), np.array([0.0, 0.0, 1.0]))
    hit1 = cube.intersection(r1)
    print("Cube Test 1:", hit1)
    assert hit1 is not None, "Cube Test 1: expected a hit"
    t1, P1, N1 = hit1
    assert abs(t1 - 2.0) < 1e-5, "Cube Test 1: t should be 2"
    assert almost_equal(P1, np.array([0.0, 0.0, -1.0])), "Cube Test 1: P incorrect"
    assert almost_equal(N1, np.array([0.0, 0.0, -1.0])), "Cube Test 1: N incorrect"

    # 2) Ray that passes next to the cube, should miss
    r2 = Ray(np.array([2.0, 2.0, -3.0]), np.array([0.0, 0.0, 1.0]))
    hit2 = cube.intersection(r2)
    print("Cube Test 2:", hit2)
    assert hit2 is None, "Cube Test 2: should be None (miss)"

    # 3) Ray starting inside the cube, going in +z → should exit at z = 1
    r3 = Ray(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    hit3 = cube.intersection(r3)
    print("Cube Test 3:", hit3)
    assert hit3 is not None, "Cube Test 3: expected a hit"
    t3, P3, N3 = hit3
    assert abs(P3[2] - 1.0) < 1e-5, "Cube Test 3: P.z should be 1"
    assert almost_equal(N3, np.array([0.0, 0.0, 1.0])), "Cube Test 3: N should be +z"

    # 4) Ray from left side aimed at center → hit left face at x = -1
    r4 = Ray(np.array([-3.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    hit4 = cube.intersection(r4)
    print("Cube Test 4:", hit4)
    assert hit4 is not None, "Cube Test 4: expected a hit"
    t4, P4, N4 = hit4
    assert abs(P4[0] + 1.0) < 1e-5, "Cube Test 4: P.x should be -1"
    assert almost_equal(N4, np.array([-1.0, 0.0, 0.0])), "Cube Test 4: N should be -x"

    # 5) Ray parallel to one face and outside bounds → no hit
    # Example: along +z, x=2, y=0 (outside x∈[-1,1])
    r5 = Ray(np.array([2.0, 0.0, -3.0]), np.array([0.0, 0.0, 1.0]))
    hit5 = cube.intersection(r5)
    print("Cube Test 5:", hit5)
    assert hit5 is None, "Cube Test 5: should be None (parallel slab outside)"

    print("All cube intersection tests ran.")

if __name__ == '__main__':
    test_cube_intersection()