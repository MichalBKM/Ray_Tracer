import argparse
from PIL import Image
import numpy as np
from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from ray import Ray
from utils import normalize, EPS
import random

def parse_scene_file(file_path):
    surfaces = []
    lights = []
    materials = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                materials.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                surfaces.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                surfaces.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                surfaces.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                lights.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, surfaces, lights, materials


def save_image(image_array, output_path):
    image = Image.fromarray(np.uint8(image_array))
    image.save(output_path)


# Find the first intersection of the ray with any of the surfaces
# Ignores surfaces is used to skip certain surfaces (e.g., the one the ray originated from)
def find_first_intersection(ray, surfaces, ignore_surfaces=None):
    if ignore_surfaces is None:
        ignore_surfaces = set()

    first_hit = None
    first_surf = None

    for surf in surfaces:
        hit = surf.intersection(ray)
        if hit is not None:
            t = hit[0]
            if t > EPS and (first_hit is None or t < first_hit[0]):
                first_hit = hit
                first_surf = surf

    return first_hit, first_surf

# Compute reflection color recursively
def get_reflection_color(ray, hit, mat, surfaces, lights, scene_settings, depth, materials):
    if np.any(mat.reflection_color):
        _, P, N = hit
        # Reflection: R = V - 2(V · N)N
        R = normalize(ray.direction - 2 * np.dot(ray.direction, N) * N)
        # New ray origin: the hit point + EPS in the reflection direction
        reflection_ray = Ray(P + EPS * R, R)
        reflection_color = (get_color_recursive(reflection_ray, depth + 1,
                            surfaces, lights, materials, scene_settings)
                            * mat.reflection_color)
    else:
        reflection_color = np.zeros(3)
    return reflection_color

# Check if the light is occluded from the hit point
def is_occluded(lgt_point, hit_point, hit_surface, surfaces):
    direction = lgt_point - hit_point
    distance_to_light = np.linalg.norm(direction)
    direction = normalize(direction)

    shadow_ray = Ray(hit_point + direction * EPS, direction)

    for surf in surfaces:
        if surf is hit_surface:
            continue

        hit = surf.intersection(shadow_ray)
        if hit is not None and EPS < hit[0] < distance_to_light - EPS:
            return True

    return False

# Calculate light intensity at the hit point considering area light and shadows
def calculate_light_intensity(lgt, hit_point, root_shadow_rays, surf, surfaces):
    # Direction from the center of the light to the hit point on the surface 
    L = hit_point - lgt.position
    L_norm = normalize(L)
    
    # Find a plane perpendicular to the ray L_norm 
    temp_vec = np.array([1, 0, 0]) if abs(L_norm[0]) < 0.9 else np.array([0, 1, 0])
    u = normalize(np.cross(L_norm, temp_vec)) # First perpendicular vector 
    v = normalize(np.cross(L_norm, u))        # Second perpendicular vector 
    
    # Setup the grid 
    N = int(root_shadow_rays)
    cell_size = lgt.radius / N 
    
    # Position of the bottom-left corner of the light rectangle
    bottom_left = lgt.position - (lgt.radius / 2) * u - (lgt.radius / 2) * v
    
    hits = 0
    total_rays = N * N 
    
    # Iterate through the NxN grid
    for i in range(N):
        for j in range(N):
            # Select a random point in each cell to avoid banding 
            random_u = random.uniform(0, 1)
            random_v = random.uniform(0, 1)
            
            # Position of the sample on the area light
            sample_point = (bottom_left + 
                           (i + random_u) * cell_size * u + 
                           (j + random_v) * cell_size * v)
            
            # Check occlusion
            # Shoot a shadow ray from the sample_point to the hit_point
            if not is_occluded(sample_point, hit_point, surf, surfaces):
                hits += 1
                
    percentage_lit = hits / total_rays
    
    intensity = (1 - lgt.shadow_intensity) + (lgt.shadow_intensity * percentage_lit)
    
    return intensity

# Local shading only: diffuse, specular, shadows
# No reflections, no recursion
def get_local_shading(ray, first_hit, surf, lights, mat, scene_settings, surfaces):
    _, P, N = first_hit

    mat_diffuse = np.array(mat.diffuse_color)
    mat_specular = np.array(mat.specular_color)

    total_diffuse = np.zeros(3)
    total_specular = np.zeros(3)

    V = normalize(np.array(-ray.direction))

    for lgt in lights:        
        # Case 1: there is a hit
        L = normalize(np.array(lgt.position) - P)
        lgt_color = np.array(lgt.color)

        # Compute Light Intensity
        light_intensity = calculate_light_intensity(lgt, P, scene_settings.root_number_shadow_rays, surf, surfaces)

        # Diffuse component
        diff = mat_diffuse * lgt_color * max(0, np.dot(N, L)) * light_intensity

        # Specular component
        R = normalize(2 * np.dot(N, L) * N - L)
        rv = np.dot(R, V)
        spec = mat_specular * lgt_color * lgt.specular_intensity * (max(rv, 0.0) ** mat.shininess)
        spec = spec * light_intensity
        total_diffuse += diff
        total_specular += spec

    return total_diffuse + total_specular

# Recursive color calculation: the main function for color computation
def get_color_recursive(ray, depth, surfaces, lights, materials, scene_settings, ignore_surfaces=None):
    if ignore_surfaces is None:
        ignore_surfaces = set()
    
    # Case 1: check recursion limit
    if depth >= scene_settings.max_recursions:
        return np.array(scene_settings.background_color)

    # Case 2: find first intersection, if none, return background color
    hit, surf = find_first_intersection(ray, surfaces, ignore_surfaces)
    if hit is None:
        return np.array(scene_settings.background_color)

    mat = materials[surf.material_index - 1]

    if mat.transparency < 1:
        local_color = get_local_shading(ray, hit, surf, lights, mat, scene_settings, surfaces)
    else:
        local_color = np.zeros(3)

    # Reflection
    reflection_color = get_reflection_color(ray, hit, mat, surfaces, lights, scene_settings, depth, materials)

    # Transparency
    transparency_color = np.zeros(3)
    if mat.transparency > 0:
        _, P, _ = hit
        transp_ray = Ray(P + 5 * EPS * ray.direction, ray.direction)

        new_ignore = set(ignore_surfaces)
        new_ignore.add(surf)

        transparency_color = get_color_recursive(transp_ray, depth + 1, surfaces, lights, materials, scene_settings, ignore_surfaces=new_ignore)

    # Combine all components
    output_color = (
        (1 - mat.transparency) * local_color
        + mat.transparency * transparency_color
        + reflection_color
        )

    return output_color


def main():
    random.seed(0)
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, surfaces, lights, materials = parse_scene_file(args.scene_file)
    
    # Prints for debugging
    print("Scene loaded successfully")
    print(f"  Surfaces: {len(surfaces)}")
    print(f"  Lights: {len(lights)}")
    print(f"  Materials: {len(materials)}")

    # Get the image size
    image_width = args.width
    image_height = args.height

    # Prints for debugging
    print(f"Rendering image of size {image_width} x {image_height}")

    aspect_ratio = image_width / image_height
    camera = Camera(camera.position, camera.look_at, camera.up_vector, camera.screen_distance, camera.screen_width, aspect_ratio)
    
    top_left = camera.screen_geometry()
    pixel_w = camera.screen_width / image_width
    pixel_h = camera.screen_height / image_height

    image_array = np.zeros((image_height, image_width, 3))
    for i in range(image_height):
        # Prints for debugging
        if i % 10 == 0:
            print(f"Rendering row {i+1} of {image_height}")
        for j in range(image_width):
            # Discover pixel's screen location
            pixel_point = (top_left + 
                camera.right * (j * pixel_w) -
                camera.up * (i * pixel_h))
            ray = Ray(camera.position, normalize(pixel_point - camera.position))
            
            # Check Intersection of the ray with all surfaces in the scene
            hit, surf = find_first_intersection(ray, surfaces)
            color = get_color_recursive(ray, 0, surfaces, lights, materials, scene_settings)
            image_array[i,j] = color
                    
    # Print for debugging
    image_array = np.clip(image_array, 0, 1)
    image_array = (image_array * 255).astype(np.uint8)
    print("Rendering finished, saving image...")
    
    # Save the output image
    save_image(image_array, args.output_image)

    # Print for debugging
    print("Image saved successfully")

    
if __name__ == '__main__':
    main()
