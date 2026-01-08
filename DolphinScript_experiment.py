# SimpleDolphinScript.py
# Slave process: runs inside Dolphin (felk/dolphin scripting)
# - Connects to master at localhost:26330
# - Receives ("ACT", a) / ("RESET", 0)
# - Applies Wiimote acceleration continuously on each frameadvance callback
# - Reads a frame, computes reward from pixel (280,446), returns obs bytes

print("SimpleDolphinScript (Just Dance) started!")

from dolphin import event, controller
from multiprocessing.connection import Client
from PIL import Image
import numpy as np


HOST = "localhost"
PORT = 26330
AUTHKEY = b"secret password"

# Observation size (small, stable for RL/debug)
OBS_W = 140
OBS_H = 75

# Judgement pixel in RAW framedrawn() resolution
PIXEL_X = 280
PIXEL_Y = 446

# -----------------------------
# Reward classifier (from our current detector)
# -----------------------------
def classify_pixel(r: int, g: int, b: int):
    if r < 45 and g < 50 and b < 50:
        return None
    if r > 120 and g < 60 and b < 60:
        return "X"          # red
    if r > 200 and g > 150 and b < 80:
        return "Yeah"       # gold/yellow
    if g > 170 and r > 100:
        return "Perfect"    # green
    if b > 200 and r < 20:
        return "Good"       # blue
    if r > 120 and b > 120:
        return "OK"         # purple
    return "Unknown"

REWARD_MAP = {
    "X": 0.0,
    "OK": 0.3,
    "Good": 0.5,
    "Perfect": 1.0,
    "Yeah": 0.8,
}

# -----------------------------
# Wiimote accel action space (27 actions)
# -----------------------------
MAX_ACCEL = 32767
BASE = 100     # tune this up if there is no response (e.g., 1000..10000)
SCALE = 1

def _clamp_int(x: float) -> int:
    return int(max(-MAX_ACCEL, min(MAX_ACCEL, round(x))))

def _v(v: float) -> int:
    return _clamp_int(v * SCALE)

def set_accel(controller_id: int, x: float, y: float, z: float):
    controller.set_wiimote_acceleration(controller_id, _v(x), _v(y), _v(z))

B = BASE
ACTION_TABLE = [
    (0, 0, 0),              # 0 idle

    (-B, 0, 0),             # 1 left
    ( B, 0, 0),             # 2 right
    (0,  B, 0),             # 3 up
    (0, -B, 0),             # 4 down
    (0, 0,  B),             # 5 forward
    (0, 0, -B),             # 6 backward

    (-B,  B, 0),            # 7 up_left
    ( B,  B, 0),            # 8 up_right
    (-B, -B, 0),            # 9 down_left
    ( B, -B, 0),            # 10 down_right

    (0,  B,  B),            # 11 forward_up
    (0, -B,  B),            # 12 forward_down
    (-B, 0,  B),            # 13 forward_left
    ( B, 0,  B),            # 14 forward_right

    (0,  B, -B),            # 15 backward_up
    (0, -B, -B),            # 16 backward_down
    (-B, 0, -B),            # 17 backward_left
    ( B, 0, -B),            # 18 backward_right

    (-B,  B,  B),           # 19 forward_up_left
    ( B,  B,  B),           # 20 forward_up_right
    (-B, -B,  B),           # 21 forward_down_left
    ( B, -B,  B),           # 22 forward_down_right

    (-B,  B, -B),           # 23 backward_up_left
    ( B,  B, -B),           # 24 backward_up_right
    (-B, -B, -B),           # 25 backward_down_left
    ( B, -B, -B),           # 26 backward_down_right
]
N_ACTIONS = len(ACTION_TABLE)  # 27


# -----------------------------
# Core state
# -----------------------------
conn = Client((HOST, PORT), authkey=AUTHKEY)

current_action = 0
current_accel = ACTION_TABLE[0]

armed = True        # edge-trigger for reward
last_label = None


def set_action(a: int):
    global current_action, current_accel
    if not (0 <= int(a) < N_ACTIONS):
        a = 0
    current_action = int(a)
    current_accel = ACTION_TABLE[current_action]


def reset_detector():
    global armed, last_label
    armed = True
    last_label = None


def process_obs(img_rgb: Image.Image) -> bytes:
    # grayscale + resize to small obs
    img = img_rgb.convert("L").resize((OBS_W, OBS_H))
    arr = np.asarray(img, dtype=np.uint8)
    return arr.tobytes()


def reward_from_frame(img_rgb: Image.Image) -> float:
    global armed, last_label

    w, h = img_rgb.size
    if not (0 <= PIXEL_X < w and 0 <= PIXEL_Y < h):
        # If this happens, our pixel coords don't match framedrawn() resolution.
        # In that case, always reward 0 and let master see trunc via info if we want.
        return 0.0

    r, g, b = img_rgb.getpixel((PIXEL_X, PIXEL_Y))
    label = classify_pixel(r, g, b)

    if label is None:
        armed = True
        return 0.0

    if armed:
        armed = False
        last_label = label
        return float(REWARD_MAP.get(label, 0.0))

    return 0.0


# Apply accel continuously every frame
def on_frame():
    x, y, z = current_accel
    set_accel(0, x, y, z)

event.on_frameadvance(on_frame)


async def read_one_frame():
    w, h, data = await event.framedrawn()
    return Image.frombytes("RGB", (w, h), data, "raw")


async def send_obs_now():
    img = await read_one_frame()
    obs_bytes = process_obs(img)
    conn.send(("OBS", obs_bytes))


async def main():
    # small settle
    for _ in range(8):
        await event.frameadvance()

    # handshake
    conn.send(("READY",))

    # initial obs (so master can reset cleanly)
    await send_obs_now()

    while True:
        msg = conn.recv()
        if not (isinstance(msg, tuple) and len(msg) >= 2):
            continue

        cmd, payload = msg[0], msg[1]

        if cmd == "RESET":
            set_action(0)
            reset_detector()
            await send_obs_now()
            continue

        if cmd == "ACT":
            set_action(int(payload))

            img = await read_one_frame()
            obs_bytes = process_obs(img)

            r = reward_from_frame(img)

            done = False   # implement later (song end)
            trunc = False  # implement later (desync / bad state)

            info = {
                "action": current_action,
                "last_label": last_label,
            }

            conn.send(("STEP", obs_bytes, float(r), bool(done), bool(trunc), info))
            continue


await main()
