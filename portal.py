import cv2
import numpy as np
import math
import random
import time


class Portal:

    def __init__(self, radius=120):
        self.x = 320
        self.y = 240
        self.radius = radius
        
        # Particle system
        self.particles = []
        
        # Multi-color themes (BGR format for OpenCV)
        self.themes = [
            {
                'name': 'Ethereal Nebula',
                'main_ring': (255, 0, 255),      # Magenta
                'inner_ring': (255, 200, 255),  # Light pink-white
                'glow': (255, 0, 128),          # Deep violet/pink
                'particles': [(255, 100, 255), (200, 50, 255), (255, 255, 255)]
            },
            {
                'name': 'Solar Flare',
                'main_ring': (0, 128, 255),     # Orange
                'inner_ring': (180, 220, 255),  # Yellow-white
                'glow': (0, 0, 255),            # Bright Red
                'particles': [(0, 200, 255), (0, 100, 255), (255, 255, 255)]
            },
            {
                'name': 'Glacial Frost',
                'main_ring': (255, 255, 0),     # Cyan
                'inner_ring': (255, 255, 200),  # Ice white
                'glow': (255, 100, 0),          # Blue-Cyan
                'particles': [(255, 200, 0), (255, 255, 255), (100, 255, 255)]
            },
            {
                'name': 'Toxic Reactor',
                'main_ring': (0, 255, 128),     # Lime Green
                'inner_ring': (200, 255, 200),  # Light Mint
                'glow': (0, 200, 0),            # Toxic Green
                'particles': [(128, 255, 0), (0, 255, 255), (255, 255, 255)]
            }
        ]
        self.theme_idx = 0

    def cycle_theme(self):
        """Cycles to the next theme and returns its name."""
        self.theme_idx = (self.theme_idx + 1) % len(self.themes)
        return self.themes[self.theme_idx]['name']

    def get_current_theme(self):
        """Returns the dictionary of the active theme."""
        return self.themes[self.theme_idx]

    # -------------------------
    # Smooth Position Update
    # -------------------------
    def update(self, x, y):
        speed = 0.18
        self.x = int(self.x + (x - self.x) * speed)
        self.y = int(self.y + (y - self.y) * speed)

    # -------------------------
    # Smooth Radius Update (Responsive Bounds)
    # -------------------------
    def set_radius(self, radius, frame_w=640):
        # Bound the radius dynamically based on frame width (e.g. 8% to 28% of width)
        min_rad = max(40, int(frame_w * 0.09))
        max_rad = max(120, int(frame_w * 0.28))
        
        radius = max(min_rad, min(max_rad, radius))

        speed = 0.20
        self.radius = int(
            self.radius + (radius - self.radius) * speed
        )

    # -------------------------
    # Draw Portal
    # -------------------------
    def draw(self, frame, background):
        h, w = frame.shape[:2]
        theme = self.themes[self.theme_idx]

        # Smooth Rounded Rectangle Mask
        mask = np.zeros((h, w), dtype=np.uint8)
        rad_w = int(self.radius * 1.2)
        rad_h = self.radius
        
        radius_corner = int(self.radius * 0.4)
        x1, y1 = self.x - rad_w, self.y - rad_h
        x2, y2 = self.x + rad_w, self.y + rad_h
        
        cv2.rectangle(mask, (x1 + radius_corner, y1), (x2 - radius_corner, y2), 255, -1)
        cv2.rectangle(mask, (x1, y1 + radius_corner), (x2, y2 - radius_corner), 255, -1)
        cv2.circle(mask, (x1 + radius_corner, y1 + radius_corner), radius_corner, 255, -1)
        cv2.circle(mask, (x2 - radius_corner, y1 + radius_corner), radius_corner, 255, -1)
        cv2.circle(mask, (x1 + radius_corner, y2 - radius_corner), radius_corner, 255, -1)
        cv2.circle(mask, (x2 - radius_corner, y2 - radius_corner), radius_corner, 255, -1)

        # Soft Edge
        mask = cv2.GaussianBlur(mask, (21, 21), 6)

        alpha = mask.astype(np.float32) / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])

        bg = background.copy().astype(np.float32)

        # Match background lighting/color to live frame
        # We sample the edges of the screen to avoid the user skewing the colors
        mask_edges = np.zeros((h, w), dtype=np.uint8)
        mask_edges[:15, :] = 255
        mask_edges[-15:, :] = 255
        mask_edges[:, :15] = 255
        mask_edges[:, -15:] = 255
        
        # Only use edges that are outside the portal
        mask_inv = (255 - (alpha[:, :, 0] * 255)).astype(np.uint8)
        mask_edges = cv2.bitwise_and(mask_edges, mask_inv)
        
        if cv2.countNonZero(mask_edges) > 100:
            mean_frame = cv2.mean(frame, mask=mask_edges)
            mean_bg = cv2.mean(background, mask=mask_edges)
            diff = np.array(mean_frame[:3]) - np.array(mean_bg[:3])
            
            # Smooth the color difference over time to avoid flickering
            if not hasattr(self, 'color_diff_smooth'):
                self.color_diff_smooth = diff
            else:
                self.color_diff_smooth = self.color_diff_smooth * 0.9 + diff * 0.1
                
            bg += self.color_diff_smooth
            bg = np.clip(bg, 0, 255)

        # Inner side color tint based on theme
        color_tint = np.array(theme['inner_ring'], dtype=np.float32)
        bg_tinted = bg * 0.85 + color_tint * 0.15

        result = (
            frame.astype(np.float32) * (1 - alpha)
            + bg_tinted * alpha
        )
        result = result.astype(np.uint8)

        # -------------------------
        # Swirling Particles removed

        return result