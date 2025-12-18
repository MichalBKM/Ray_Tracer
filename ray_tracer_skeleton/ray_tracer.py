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
from utils import normalize

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


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")

# find the first intersection of the ray with any of the surfaces
def find_first_intersection(ray, surfaces):
    first_hit = None
    first_surf = None
    for surf in surfaces:
                hit = surf.intersection(ray)
                if hit is not None:
                    t = hit[0]
                    if t < 0:
                        continue
                    elif  first_hit is None or t < first_hit[0]:
                        first_hit = hit
                        first_surf = surf

    return first_hit, first_surf

def compute_color(ray, first_hit, surf, lights, mat, scene_settings):
    _, P, N = first_hit
    ambient = scene_settings.background_color * mat.diffuse_color
    diffuse, specular = 0, 0
    V = -ray.direction 

    for lgt in lights:
        #Does the light hit the surface?
        
        #Case_1: there is a hit
        L = normalize(lgt.position - P)
        diffuse += mat.diffuse_color * lgt.color * max(0, np.dot(N, L))
        R = normalize(2 * np.dot(N, L) * N - L)
        specular += mat.specular_color * lgt.color * max(0, np.dot(R, V)) ** mat.shininess
        color += (diffuse + specular)

        #Case_2: there is no hit
    
    # TODO: Check if ambient = background color
    color = (ambient * mat.transparency) + (diffuse + specular) * (1 - mat.transparency) + mat.reflection_color
    return color


    
def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, surfaces, lights, materials = parse_scene_file(args.scene_file)
    
    # get the image size
    image_width = args.width
    image_height = args.height
    aspect_ratio = image_width / image_height
    camera = Camera(camera.position, camera.look_at, camera.up_vector, camera.screen_distance, camera.screen_width, aspect_ratio)
    
    
    # TODO: Implement the ray tracer
    top_left = camera.screen_geometry()
    pixel_h= camera.screen_width / image_width
    pixel_w = camera.screen_height / image_height
    
    for i in range(image_height):
        for j in range(image_width):
            # Discover pixel's screen location
            pixel_point = (top_left + 
                           camera.right * (j * pixel_w + pixel_w / 2) -
                           camera.up * (i * pixel_h + pixel_h / 2))
            ray = Ray(pixel_point, camera)
            
            #Check Intersection of the ray with all surfaces in the scene
            hit, surf = find_first_intersection(ray, surfaces)
            if hit is None or surf is None:
                continue
            
            # TODO: Compute the color of the pixel
            material = materials[surf.material_index - 1]
            color = compute_color(ray, hit, surf, lights, material, scene_settings)
            
                    
 


    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)

    
if __name__ == '__main__':
    main()
