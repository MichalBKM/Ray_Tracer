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
    
def parse_scene_file(file_path):
    objects = []
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
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")

def get_ray(pixel_point, camera):
    origin = camera.position
    direction = pixel_point - origin
    direction = direction / np.linalg.norm(direction)
    return origin, direction
    
def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    
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
            pixel_point = (top_left + 
                           camera.right * (j * pixel_w + pixel_w / 2) -
                           camera.up * (i * pixel_h + pixel_h / 2))
            origin, direction = get_ray(pixel_point, camera)
            ray = Ray(origin, direction)
            for obj in objects:
                if isinstance(obj, Sphere):
                    t, P, N = obj.intersection(ray)
                    print(t, P, N)
                    break
            break


    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
