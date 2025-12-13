import numpy as np
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from ray import Ray  # your Ray(pixel_point, camera)

def almost_equal(a, b, tol=1e-5):
    return np.allclose(a, b, atol=tol)

class DummyCamera:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)

def test_plane_intersection():
    plane = InfinitePlane(np.array([0.0, 1.0, 0.0]), -1.0, 0)  # y = 1
    cam = DummyCamera([0.0, 0.0, 0.0])

    # 1) From below, up: choose pixel_point along +y
    r1 = Ray(pixel_point=np.array([0.0, 1.0, 0.0]), camera=cam)  # direction (0,1,0)
    hit1 = plane.intersection(r1)
    print("Plane Test 1:", hit1)
    assert hit1 is not None, "Plane Test 1: expected a hit"
    t1, P1, N1 = hit1
    assert abs(t1 - 1.0) < 1e-5, "Plane Test 1: t should be 1"
    assert almost_equal(P1, np.array([0.0, 1.0, 0.0])), "Plane Test 1: P incorrect"
    assert almost_equal(N1, np.array([0.0, 1.0, 0.0])), "Plane Test 1: N incorrect"

    # 2) Parallel above: set camera above plane, shoot along +x
    cam2 = DummyCamera([0.0, 2.0, 0.0])
    r2 = Ray(pixel_point=np.array([1.0, 2.0, 0.0]), camera=cam2)  # direction (1,0,0)
    hit2 = plane.intersection(r2)
    print("Plane Test 2:", hit2)
    assert hit2 is None, "Plane Test 2: should be None (parallel and above)"

    # 3) Above and pointing up: plane is below, so t < 0 -> None
    cam3 = DummyCamera([0.0, 2.0, 0.0])
    r3 = Ray(pixel_point=np.array([0.0, 3.0, 0.0]), camera=cam3)  # direction (0,1,0)
    hit3 = plane.intersection(r3)
    print("Plane Test 3:", hit3)
    assert hit3 is None, "Plane Test 3: should be None (plane behind ray direction)"

    # 4) Starting on the plane: camera exactly on y=1, shoot up
    cam4 = DummyCamera([0.0, 1.0, 0.0])
    r4 = Ray(pixel_point=np.array([0.0, 2.0, 0.0]), camera=cam4)  # direction (0,1,0)
    hit4 = plane.intersection(r4)
    print("Plane Test 4:", hit4)
    print("All plane intersection tests ran.")

def test_cube_intersection():
    cube = Cube(np.array([0.0, 0.0, 0.0]), 2.0, 0)  # bounds [-1,1] on all axes

    # 1) From z=-3 towards +z: hit at (0,0,-1)
    cam1 = DummyCamera([0.0, 0.0, -3.0])
    r1 = Ray(pixel_point=np.array([0.0, 0.0, -2.0]), camera=cam1)  # direction (0,0,1)
    hit1 = cube.intersection(r1)
    print("Cube Test 1:", hit1)
    assert hit1 is not None, "Cube Test 1: expected a hit"
    t1, P1, N1 = hit1
    assert abs(t1 - 2.0) < 1e-5, "Cube Test 1: t should be 2"
    assert almost_equal(P1, np.array([0.0, 0.0, -1.0])), "Cube Test 1: P incorrect"
    assert almost_equal(N1, np.array([0.0, 0.0, -1.0])), "Cube Test 1: N incorrect"

    # 2) Miss: camera at (2,2,-3) shoot +z
    cam2 = DummyCamera([2.0, 2.0, -3.0])
    r2 = Ray(pixel_point=np.array([2.0, 2.0, -2.0]), camera=cam2)  # direction (0,0,1)
    hit2 = cube.intersection(r2)
    print("Cube Test 2:", hit2)
    assert hit2 is None, "Cube Test 2: should be None (miss)"

    # 3) Starting inside at origin, shoot +z: exit at z=1
    cam3 = DummyCamera([0.0, 0.0, 0.0])
    r3 = Ray(pixel_point=np.array([0.0, 0.0, 1.0]), camera=cam3)  # direction (0,0,1)
    hit3 = cube.intersection(r3)
    print("Cube Test 3:", hit3)
    assert hit3 is not None, "Cube Test 3: expected a hit"
    t3, P3, N3 = hit3
    assert abs(P3[2] - 1.0) < 1e-5, "Cube Test 3: P.z should be 1"
    assert almost_equal(N3, np.array([0.0, 0.0, 1.0])), "Cube Test 3: N should be +z"

    # 4) From left x=-3 towards +x: hit at x=-1
    cam4 = DummyCamera([-3.0, 0.0, 0.0])
    r4 = Ray(pixel_point=np.array([-2.0, 0.0, 0.0]), camera=cam4)  # direction (1,0,0)
    hit4 = cube.intersection(r4)
    print("Cube Test 4:", hit4)
    assert hit4 is not None, "Cube Test 4: expected a hit"
    t4, P4, N4 = hit4
    assert abs(P4[0] + 1.0) < 1e-5, "Cube Test 4: P.x should be -1"
    assert almost_equal(N4, np.array([-1.0, 0.0, 0.0])), "Cube Test 4: N should be -x"

    # 5) Parallel slab outside: x=2, shoot +z, should miss
    cam5 = DummyCamera([2.0, 0.0, -3.0])
    r5 = Ray(pixel_point=np.array([2.0, 0.0, -2.0]), camera=cam5)  # direction (0,0,1)
    hit5 = cube.intersection(r5)
    print("Cube Test 5:", hit5)
    assert hit5 is None, "Cube Test 5: should be None (parallel slab outside)"

    print("All cube intersection tests ran.")

if __name__ == "__main__":
    test_plane_intersection()
    test_cube_intersection()
