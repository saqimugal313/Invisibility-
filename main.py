import cv2
import numpy as np
import time
import math
import random
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from portal import Portal

# ============================================================
#  MediaPipe Hand Connections
# ============================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# ============================================================
#  HUD Helpers
# ============================================================

def get_hand_distance(landmarks):
    """Return thumb-to-index distance for a hand landmark list."""
    tip = landmarks[8]
    thumb = landmarks[4]
    return math.hypot(tip.x - thumb.x, tip.y - thumb.y)


def scale(val, ref=1280):
    """Return val scaled relative to a 1280-wide reference frame."""
    return val  # will be used with per-frame w below

def draw_text_shadow(img, text, pos, font, fscale, color, thick,
                     shadow=(0, 0, 0), shadow_offset=2):
    """Crisp text with a multi-directional shadow for readability."""
    x, y = pos
    so = shadow_offset
    for dx, dy in [(-so, -so), (so, -so), (-so, so), (so, so),
                   (0, -so), (0, so), (-so, 0), (so, 0)]:
        cv2.putText(img, text, (x + dx, y + dy), font, fscale,
                    shadow, thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, fscale, color, thick, cv2.LINE_AA)


def draw_glass_panel(img, x1, y1, x2, y2, bg_color=(10, 10, 30),
                     alpha=0.55, border_color=None, border_thick=1, radius=8):
    """Draw a rounded-rectangle glassmorphism panel."""
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    overlay = img.copy()
    # Fill rounded rect
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), bg_color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), bg_color, -1)
    for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
        cv2.circle(overlay, (cx, cy), radius, bg_color, -1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

    # Border
    if border_color:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), border_color, border_thick)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), border_color, border_thick)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(img, (cx, cy), radius, border_color, border_thick)


def draw_custom_hand(frame, landmarks, theme):
    """Glowing hand skeleton overlay."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    line_color  = theme['main_ring']
    joint_inner = theme['inner_ring']
    joint_outer = theme['main_ring']
    glow_color  = theme['glow']

    line_thick = max(1, int(w / 480))
    outer_r    = max(4, int(w / 160))
    inner_r    = max(2, int(w / 320))

    # Glow lines
    for conn in HAND_CONNECTIONS:
        s, e = conn
        slm, elm = landmarks[s], landmarks[e]
        pt1 = (int(slm.x * w), int(slm.y * h))
        pt2 = (int(elm.x * w), int(elm.y * h))
        cv2.line(overlay, pt1, pt2, glow_color, line_thick + 3, cv2.LINE_AA)

    alpha_glow = 0.3
    cv2.addWeighted(overlay, alpha_glow, frame, 1 - alpha_glow, 0, dst=frame)

    # Solid lines
    for conn in HAND_CONNECTIONS:
        s, e = conn
        slm, elm = landmarks[s], landmarks[e]
        pt1 = (int(slm.x * w), int(slm.y * h))
        pt2 = (int(elm.x * w), int(elm.y * h))
        cv2.line(frame, pt1, pt2, line_color, line_thick, cv2.LINE_AA)

    # Joints
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), outer_r, joint_outer, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), inner_r, joint_inner, -1, cv2.LINE_AA)


# ============================================================
#  Camera
# ============================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

# ============================================================
#  MediaPipe Tasks Setup
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at: {MODEL_PATH}")
    print("Download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
          "hand_landmarker/float16/latest/hand_landmarker.task")
    cap.release()
    exit()

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
hand_landmarker = HandLandmarker.create_from_options(options)

# ============================================================
#  Window
# ============================================================
window_name = "AI Magic Invisibility Portal  |  Developed by Azhar Khan"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# ============================================================
#  Background Capture with Countdown
# ============================================================
print("Stand away from the camera for background capture...")

ret, init_frame = cap.read()
if not ret:
    print("ERROR: Cannot read camera for background setup")
    cap.release()
    exit()

init_frame = cv2.flip(init_frame, 1)
h_i, w_i = init_frame.shape[:2]

# Responsive scales for countdown screen
fs_cd  = max(0.5, w_i / 900.0)
fth_cd = max(1, int(w_i / 600))

for countdown in range(3, 0, -1):
    cd_frame = init_frame.copy()

    # Dark overlay
    dark = cd_frame.copy()
    cv2.rectangle(dark, (0, 0), (w_i, h_i), (5, 5, 20), -1)
    cv2.addWeighted(dark, 0.65, cd_frame, 0.35, 0, dst=cd_frame)

    # Pulsing ring
    pulse = int(math.sin(time.time() * 6) * 6)
    cv2.circle(cd_frame, (w_i // 2, h_i // 2), 70 + pulse, (0, 220, 255), 3, cv2.LINE_AA)
    cv2.circle(cd_frame, (w_i // 2, h_i // 2), 50 + pulse, (0, 120, 200), 2, cv2.LINE_AA)

    # Countdown number (centered)
    num_text = str(countdown)
    (tw, th), _ = cv2.getTextSize(num_text, cv2.FONT_HERSHEY_DUPLEX, fs_cd * 3.5, fth_cd + 2)
    cx = (w_i - tw) // 2
    cy = (h_i + th) // 2
    draw_text_shadow(cd_frame, num_text, (cx, cy),
                     cv2.FONT_HERSHEY_DUPLEX, fs_cd * 3.5, (0, 220, 255), fth_cd + 2)

    # Labels
    draw_text_shadow(cd_frame, "CAPTURING BACKGROUND",
                     (int(w_i * 0.22), int(h_i * 0.20)),
                     cv2.FONT_HERSHEY_DUPLEX, fs_cd * 1.0, (255, 255, 255), fth_cd + 1)
    draw_text_shadow(cd_frame, "Please step out of frame",
                     (int(w_i * 0.26), int(h_i * 0.80)),
                     cv2.FONT_HERSHEY_SIMPLEX, fs_cd * 0.85, (180, 230, 255), fth_cd)

    cv2.imshow(window_name, cd_frame)
    cv2.waitKey(1000)

ret, background = cap.read()
if not ret:
    print("ERROR: Failed to capture background.")
    cap.release()
    exit()

background = cv2.flip(background, 1)
print("Background captured successfully!")

# ============================================================
#  Main Loop State
# ============================================================
portal             = Portal(radius=120)
show_hand_skeleton = True
prev_time          = time.time()
right_zoom_prev    = None
left_zoom_prev     = None

# ============================================================
#  Main Loop
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # ---- FPS ----
    curr_time = time.time()
    fps = int(1.0 / max(curr_time - prev_time, 1e-6))
    prev_time = curr_time

    # ---- Responsive scale factors ----
    ref      = 1280.0
    rs       = w / ref          # ratio to 1280-wide reference
    fscale_L = max(0.55, rs * 1.15)   # large  (title)
    fscale_M = max(0.45, rs * 0.90)   # medium (dev / status)
    fscale_S = max(0.35, rs * 0.72)   # small  (controls)
    fth_L    = max(1, int(rs * 2.4))
    fth_M    = max(1, int(rs * 1.8))
    fth_S    = max(1, int(rs * 1.4))

    mx = int(w * 0.025)          # left margin
    my = int(h * 0.035)          # top margin
    sp = int(h * 0.048)          # line spacing

    # ---- Hand tracking ----
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect(mp_img)

    if result.hand_landmarks:
        right_hand = None
        left_hand = None

        for idx, lms in enumerate(result.hand_landmarks):
            handedness = "Unknown"
            if idx < len(result.handedness) and result.handedness[idx]:
                handedness = result.handedness[idx][0].category_name

            if handedness == "Right":
                right_hand = lms
            elif handedness == "Left":
                left_hand = lms

        active_hand = right_hand or left_hand
        if active_hand is not None:
            tip = active_hand[8]
            portal.update(int(tip.x * w), int(tip.y * h))

            if show_hand_skeleton:
                draw_custom_hand(frame, active_hand, portal.get_current_theme())

        if right_hand is not None:
            dist = get_hand_distance(right_hand)
            if right_zoom_prev is None:
                right_zoom_prev = dist
            else:
                delta = (dist - right_zoom_prev) * 1400
                portal.set_radius(portal.radius + int(delta), w)
                right_zoom_prev = dist
        else:
            right_zoom_prev = None

        if left_hand is not None:
            dist = get_hand_distance(left_hand)
            if left_zoom_prev is None:
                left_zoom_prev = dist
            else:
                delta = (dist - left_zoom_prev) * 1400
                portal.set_radius(portal.radius - int(delta), w)
                left_zoom_prev = dist
        else:
            left_zoom_prev = None

    # ---- Portal render ----
    frame = portal.draw(frame, background)
    theme = portal.get_current_theme()

    # ============================================================
    #  HUD  –  Top-left panel
    # ============================================================
    # Measure texts to size the panel
    title_text  = "AI Magic Invisibility Portal"
    dev_text    = "\u2665  Developed by  Azhar Khan"
    status_text = f"Theme: {theme['name']}   |   FPS: {fps:3d}"

    (tw_t, th_t), _ = cv2.getTextSize(title_text,  cv2.FONT_HERSHEY_DUPLEX,  fscale_L, fth_L)
    (tw_d, th_d), _ = cv2.getTextSize(dev_text,    cv2.FONT_HERSHEY_DUPLEX,  fscale_M, fth_M)
    (tw_s, th_s), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, fscale_S, fth_S)

    pad   = int(h * 0.018)
    panel_w = max(tw_t, tw_d, tw_s) + pad * 2
    panel_h = pad + (th_t + sp) + (th_d + sp) + (th_s + pad)

    # Glass panel background
    draw_glass_panel(frame,
                     mx - pad, my - pad,
                     mx + panel_w, my + panel_h,
                     bg_color=(8, 8, 28), alpha=0.62,
                     border_color=theme['main_ring'], border_thick=1, radius=10)

    # Thin accent line under title
    accent_y = my + th_t + int(sp * 0.4)
    cv2.line(frame,
             (mx, accent_y), (mx + panel_w - pad, accent_y),
             theme['glow'], 1, cv2.LINE_AA)

    # Title  – bright white
    draw_text_shadow(frame, title_text, (mx, my + th_t),
                     cv2.FONT_HERSHEY_DUPLEX, fscale_L, (240, 245, 255), fth_L,
                     shadow=(0, 0, 0), shadow_offset=2)

    # Developer  – theme accent colour with a warm gold tint
    dev_color = (80, 200, 255)   # gold-ish in BGR
    draw_text_shadow(frame, dev_text,
                     (mx, my + th_t + sp + th_d),
                     cv2.FONT_HERSHEY_DUPLEX, fscale_M, dev_color, fth_M,
                     shadow=(0, 0, 0), shadow_offset=1)

    # Status
    draw_text_shadow(frame, status_text,
                     (mx, my + th_t + sp + th_d + sp + th_s),
                     cv2.FONT_HERSHEY_SIMPLEX, fscale_S, theme['inner_ring'], fth_S,
                     shadow=(0, 0, 0), shadow_offset=1)

    # ============================================================
    #  HUD  –  Bottom controls bar
    # ============================================================
    ctrl_text = "[C] Cycle Theme    [H] Toggle Hand    [B] Recapture BG    [S] Screenshot    [Q] Quit    |    Right hand zoom in / Left hand zoom out"
    (tw_c, th_c), _ = cv2.getTextSize(ctrl_text, cv2.FONT_HERSHEY_SIMPLEX, fscale_S, fth_S)

    bar_pad = int(h * 0.015)
    bar_y1  = h - th_c - bar_pad * 3
    bar_y2  = h - bar_pad

    # Full-width glassmorphism bottom bar
    draw_glass_panel(frame,
                     0, bar_y1,
                     w - 1, bar_y2,
                     bg_color=(8, 8, 28), alpha=0.55,
                     border_color=None, border_thick=0, radius=0)

    # Thin top border on bar
    cv2.line(frame, (0, bar_y1), (w, bar_y1), theme['glow'], 1, cv2.LINE_AA)

    ctrl_x = (w - tw_c) // 2
    ctrl_y = bar_y1 + bar_pad + th_c
    draw_text_shadow(frame, ctrl_text, (ctrl_x, ctrl_y),
                     cv2.FONT_HERSHEY_SIMPLEX, fscale_S, (210, 225, 240), fth_S,
                     shadow=(0, 0, 0), shadow_offset=1)

    # ============================================================
    #  Show
    # ============================================================
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF

    # ---- Key handlers ----
    if key in (ord('c'), ord('C')):
        name = portal.cycle_theme()
        print(f"Theme -> {name}")

    elif key in (ord('h'), ord('H')):
        show_hand_skeleton = not show_hand_skeleton
        print(f"Hand skeleton: {show_hand_skeleton}")

    elif key in (ord('b'), ord('B')):
        print("Stand away from camera...")
        freeze = frame.copy()
        hf, wf = freeze.shape[:2]

        dark2 = freeze.copy()
        cv2.rectangle(dark2, (0, 0), (wf, hf), (5, 5, 20), -1)
        cv2.addWeighted(dark2, 0.5, freeze, 0.5, 0, dst=freeze)

        for countdown in range(3, 0, -1):
            tmp = freeze.copy()
            draw_text_shadow(tmp, "RE-CAPTURING BACKGROUND",
                             (int(wf * 0.18), int(hf * 0.45)),
                             cv2.FONT_HERSHEY_DUPLEX, fscale_L * 1.0,
                             (0, 220, 255), fth_L)
            draw_text_shadow(tmp, f"Step out of frame... {countdown}",
                             (int(wf * 0.28), int(hf * 0.58)),
                             cv2.FONT_HERSHEY_SIMPLEX, fscale_M,
                             (255, 255, 255), fth_M)
            cv2.imshow(window_name, tmp)
            cv2.waitKey(1000)

        ret2, bg2 = cap.read()
        if ret2:
            background = cv2.flip(bg2, 1)
            print("Background updated!")

    elif key in (ord('s'), ord('S')):
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")
        ts = time.strftime("%Y%m%d-%H%M%S")
        filename = f"screenshots/portal_{ts}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved to {filename}")

    elif key in (ord('q'), ord('Q')):
        break

# ============================================================
#  Cleanup
# ============================================================
hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()