import numpy as np

class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = position
        self.look_at = look_at
        self.up_vector = up_vector
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        self.setop_camera_axes()
    
    def setup_camera_axes(self):
        # FORWARD direction: calculation + normalization
        self.forward = (self.look_at - self.position)
        self.forward = self.forward / np.linalg(self.forward)

        # RIGHT direction: calculation + normalization
        self.right = np.cross(self.forward, self.up_vector)
        self.right = self.right / np.linalg(self.right)

        # UP direction: calculation + normalization
        self.up = np.cross(self.right, self.forward)
        self.up = self.up / np.linalg.norm(self.up)

