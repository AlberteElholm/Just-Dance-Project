import math
from dolphin import event, controller
from multiprocessing.connection import Client

PORT = 26330
AUTHKEY = b"secret password"

conn = Client(("localhost", PORT), authkey=AUTHKEY)
conn.send("READY")

cmd, payload = conn.recv()

frames_per_action, base = payload
  # desired acceleration vector length (magnitude)

# -------------------------
# Acceleration helpers
# -------------------------
def set_accel(controller_id: int, x: float, y: float, z: float):
    controller.set_wiimote_acceleration(controller_id, x, y, z)

def _set_dir(controller_id: int, dx: float, dy: float, dz: float):
    """Scale (dx,dy,dz) to have length == base (unless it's the zero vector)."""
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm == 0:
        set_accel(controller_id, 0, 0, 0)
        return
    s = base / norm
    set_accel(controller_id, dx * s, dy * s, dz * s)

# -------------------------
# Actions (ESTABLISHED NAMES)
# -------------------------
def idle():          set_accel(0, 0, 0, 0)

# 1D
def left():          _set_dir(0, -1, 0, 0)
def right():         _set_dir(0,  1, 0, 0)
def up():            _set_dir(0,  0, 1, 0)
def down():          _set_dir(0,  0,-1, 0)
def forward():       _set_dir(0,  0, 0, 1)
def backward():      _set_dir(0,  0, 0,-1)

# 2D
def up_left():       _set_dir(0, -1,  1, 0)
def up_right():      _set_dir(0,  1,  1, 0)
def down_left():     _set_dir(0, -1, -1, 0)
def down_right():    _set_dir(0,  1, -1, 0)

def forward_up():    _set_dir(0,  0,  1,  1)
def forward_down():  _set_dir(0,  0, -1,  1)
def forward_left():  _set_dir(0, -1,  0,  1)
def forward_right(): _set_dir(0,  1,  0,  1)

def backward_up():   _set_dir(0,  0,  1, -1)
def backward_down(): _set_dir(0,  0, -1, -1)
def backward_left(): _set_dir(0, -1,  0, -1)
def backward_right():_set_dir(0,  1,  0, -1)

# 3D
def forward_up_left():    _set_dir(0, -1,  1,  1)
def forward_up_right():   _set_dir(0,  1,  1,  1)
def forward_down_left():  _set_dir(0, -1, -1,  1)
def forward_down_right(): _set_dir(0,  1, -1,  1)

def backward_up_left():   _set_dir(0, -1,  1, -1)
def backward_up_right():  _set_dir(0,  1,  1, -1)
def backward_down_left(): _set_dir(0, -1, -1, -1)
def backward_down_right():_set_dir(0,  1, -1, -1)

ACTIONS = {
    0: idle,

    1: left,  2: right,  3: up,   4: down,  5: forward,  6: backward,

    7: up_left,    8: up_right,    9: down_left,    10: down_right,
    11: forward_up, 12: forward_down, 13: forward_left, 14: forward_right,
    15: backward_up, 16: backward_down, 17: backward_left, 18: backward_right,

    19: forward_up_left,   20: forward_up_right,
    21: forward_down_left, 22: forward_down_right,
    23: backward_up_left,  24: backward_up_right,
    25: backward_down_left,26: backward_down_right,
}

# -------------------------
# Buttons / pointer helpers
# -------------------------
def a_press():
    controller.set_wiimote_buttons(0, {"A": True})

def a_release():
    controller.set_wiimote_buttons(0, {"A": False})

def b_press():
    controller.set_wiimote_buttons(0, {"B": True})

def b_release():
    controller.set_wiimote_buttons(0, {"B": False})

async def wait_frames(n: int):
    for _ in range(n):
        await event.frameadvance()

def set_pointer():
    controller.set_wiimote_pointer(0, -0.7, -0.9)

async def a_sequence(hold_frames: int, repeats: int):
    for _ in range(repeats):
        # keep pointer steady while waiting
        for _ in range(hold_frames):
            await event.frameadvance()
            set_pointer()

        # two quick A taps with pointer set
        for _ in range(2):
            set_pointer()
            a_press()
            await event.frameadvance()

        a_release()
        await event.frameadvance()

# -------------------------
# Main loop
# -------------------------
startup = 0
episode_count = 0

while True:
    cmd, payload = conn.recv()

    if cmd == "send":
        if controller.get_wiimote_buttons(0).get("B", False) and startup < 1:
            a_press()
            b_press()
            startup += 1
            await event.frameadvance()
            conn.send(("Dancing", controller.get_wiimote_buttons(0)))
            a_release()
            b_release()

        if payload in ACTIONS:
            for _ in range(frames_per_action):
                await event.frameadvance()
                ACTIONS[payload]()  # now always magnitude==base for 2D/3D too
            conn.send(("Dancing", controller.get_wiimote_buttons(0)))
            continue

    if cmd == "reset":
        await wait_frames(1000)

        if episode_count < 2:
            episode_count += 1
            await a_sequence(hold_frames=200, repeats=2)
        else:
            await a_sequence(hold_frames=30, repeats=1)

        await wait_frames(100)

        a_press(); b_press()
        await event.frameadvance()
        a_press(); b_press()
        conn.send(("Dancing", controller.get_wiimote_buttons(0)))
