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

        # Rectangle Mask
        mask = np.zeros((h, w), dtype=np.uint8)
        rad_w = int(self.radius * 1.2)
        rad_h = self.radius
        cv2.rectangle(
            mask,
            (self.x - rad_w, self.y - rad_h),
            (self.x + rad_w, self.y + rad_h),
            255,
            -1
        )

        # Soft Edge
        mask = cv2.GaussianBlur(mask, (15, 15), 4)

        alpha = mask.astype(np.float32) / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])

        bg = background.copy()

        result = (
            frame.astype(np.float32) * (1 - alpha)
            + bg.astype(np.float32) * alpha
        )
        result = result.astype(np.uint8)

        # -------------------------
        # Dynamic Pulsing Glow
        # -------------------------
        glow = result.copy()
        pulse = math.sin(time.time() * 8.0) * 6.0
        glow_color = theme['glow']

        current_r = int(self.radius + 6 + pulse)
        curr_w = int(current_r * 1.2)
        curr_h = current_r
        cv2.rectangle(
            glow,
            (self.x - curr_w, self.y - curr_h),
            (self.x + curr_w, self.y + curr_h),
            glow_color,
            2,
            cv2.LINE_AA
        )

        result = cv2.addWeighted(
            glow,
            0.35,
            result,
            0.65,
            0
        )

        # -------------------------
        # Swirling Particles
        # -------------------------
        if len(self.particles) < 60:
            for _ in range(random.randint(1, 2)):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(0.04, 0.12) * random.choice([-1, 1])
                offset = random.uniform(-8, 8)
                life = 1.0
                color = random.choice(theme['particles'])
                size = random.randint(2, 5)
                self.particles.append({
                    'angle': angle,
                    'speed': speed,
                    'offset': offset,
                    'life': life,
                    'color': color,
                    'size': size
                })

        active_particles = []
        for p in self.particles:
            p['angle'] += p['speed']
            p['life'] -= 0.035
            p['offset'] += random.uniform(-0.2, 0.5)
            
            if p['life'] > 0:
                active_particles.append(p)
                r = self.radius + p['offset']
                px = int(self.x + (r * 1.2) * math.cos(p['angle']))
                py = int(self.y + r * math.sin(p['angle']))
                
                if 0 <= px < w and 0 <= py < h:
                    sz = int(p['size'] * p['life'])
                    if sz > 0:
                        cv2.rectangle(result, (px - sz, py - sz), (px + sz, py + sz), p['color'], -1, cv2.LINE_AA)
                        cv2.rectangle(result, (px - sz - 1, py - sz - 1), (px + sz + 1, py + sz + 1), p['color'], 1, cv2.LINE_AA)
                        
        self.particles = active_particles

        # -------------------------
        # Portal Outline
        # -------------------------
        rad_w = int(self.radius * 1.2)
        rad_h = self.radius

        cv2.rectangle(
            result,
            (self.x - rad_w, self.y - rad_h),
            (self.x + rad_w, self.y + rad_h),
            theme['main_ring'],
            2,
            cv2.LINE_AA
        )

        return result