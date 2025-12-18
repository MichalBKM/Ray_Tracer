import numpy as np
from utils import normalize

class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width, aspect_ratio=1.0):
        self.position = position
        self.look_at = look_at
        self.up_vector = up_vector
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        self.screen_height = self.screen_width / aspect_ratio
        self.setup_camera_axes()

    
    def setup_camera_axes(self):
        # FORWARD direction: calculation + normalization
        self.forward = np.array(self.look_at) - np.array(self.position)
        self.forward = normalize(self.forward) 

        # RIGHT direction: calculation + normalization
        self.right = np.cross(self.forward, self.up_vector)
        self.right = normalize(self.right)

        # UP direction: calculation + normalization
        self.up = np.cross(self.right, self.forward)
        self.up = normalize(self.up)
    
    def screen_geometry(self):
        pos_array = np.array(self.position)
        screen_center = pos_array + self.forward * self.screen_distance

        half_w = self.screen_width / 2
        half_h = self.screen_height / 2

        top_left = screen_center + (self.up * half_h) - (self.right * half_w)

        return top_left

