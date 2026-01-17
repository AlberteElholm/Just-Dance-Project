import math
from dolphin import event, controller
from multiprocessing.connection import Client

PORT = 26330
AUTHKEY = b"secret password"

conn = Client(("localhost", PORT), authkey=AUTHKEY)
conn.send("READY")

frames_per_action = 4
base = 100.0  # REQUIRED final magnitude

# -------------------------
# Acceleration helpers (always final magnitude == base)
# -------------------------
def set_accel(controller_id: int, x: float, y: float, z: float):
    controller.set_wiimote_acceleration(controller_id, x, y, z)

def _norm3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)

def _normalize_to_base(x: float, y: float, z: float):
    n = _norm3(x, y, z)
    if n == 0:
        return 0.0, 0.0, 0.0
    s = base / n
    return x * s, y * s, z * s

def _set_dir(controller_id: int, dx: float, dy: float, dz: float):
    x, y, z = _normalize_to_base(dx, dy, dz)
    set_accel(controller_id, x, y, z)

# -------------------------
# Actions (ESTABLISHED NAMES, 1D only)
# -------------------------
def idle():          set_accel(0, 0, 0, 0)

def left():          _set_dir(0, -1, 0, 0)
def right():         _set_dir(0,  1, 0, 0)
def up():            _set_dir(0,  0, 1, 0)
def down():          _set_dir(0,  0,-1, 0)
def forward():       _set_dir(0,  0, 0, 1)
def backward():      _set_dir(0,  0, 0,-1)

# Map 0..6 (idle + 6 directions)
ACTIONS = {
    0: idle,
    1: left,  2: right,  3: up,  4: down,  5: forward,  6: backward,
}

# For send2 composition: translate action id -> unit direction (not scaled)
# (idle returns 0-vector)
DIRS = {
    0: (0.0,  0.0,  0.0),
    1: (-1.0, 0.0,  0.0),
    2: (1.0,  0.0,  0.0),
    3: (0.0,  1.0,  0.0),
    4: (0.0, -1.0,  0.0),
    5: (0.0,  0.0,  1.0),
    6: (0.0,  0.0, -1.0),
}

def do_send(a_id: int):
    # Single-vector mode: magnitude==base (or zero for idle)
    fn = ACTIONS.get(a_id, idle)
    fn()

def do_send2(base_id: int, add_id: int):
    # Two-vector mode:
    #   - each component vector is normalized to base (unless idle)
    #   - sum them
    #   - normalize the SUM to base (so final magnitude always base)
    bx, by, bz = DIRS.get(int(base_id), (0.0, 0.0, 0.0))
    ax, ay, az = DIRS.get(int(add_id),  (0.0, 0.0, 0.0))

    # normalize each to base (idle stays zero)
    bx, by, bz = _normalize_to_base(bx, by, bz)
    ax, ay, az = _normalize_to_base(ax, ay, az)

    sx, sy, sz = (bx + ax), (by + ay), (bz + az)

    # normalize final sum to base (or zero if sum is zero)
    sx, sy, sz = _normalize_to_base(sx, sy, sz)

    set_accel(0, sx, sy, sz)

# -------------------------
# Buttons / pointer helpers (unchanged)
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
# Main loop (supports "send" and "send2")
# -------------------------
startup = 0
episode_count = 0

while True:
    cmd, payload = conn.recv()

    if cmd == "send":
        # Startup A+B once (same behavior you had)
        if controller.get_wiimote_buttons(0).get("B", False) and startup < 1:
            a_press()
            b_press()
            startup += 1
            await event.frameadvance()
            conn.send(("Dancing", controller.get_wiimote_buttons(0)))
            a_release()
            b_release()

        # payload is an int action id (0..6)
        a_id = int(payload)
        for _ in range(frames_per_action):
            await event.frameadvance()
            do_send(a_id)
        conn.send(("Dancing", controller.get_wiimote_buttons(0)))
        continue

    if cmd == "send2":
        # payload is (base_id, add_id), each 0..6
        base_id, add_id = payload
        base_id = int(base_id)
        add_id  = int(add_id)

        for _ in range(frames_per_action):
            await event.frameadvance()
            do_send2(base_id, add_id)
        conn.send(("Dancing", controller.get_wiimote_buttons(0)))
        continue

    if cmd == "reset":
        await wait_frames(800)

        if episode_count < 2:
            episode_count += 1
            await a_sequence(hold_frames=200, repeats=2)
        else:
            await a_sequence(hold_frames=30, repeats=1)

        await wait_frames(100)

        # Signal master that reset finished (A+B pressed)
        a_press(); b_press()
        await event.frameadvance()
        a_press(); b_press()
        conn.send(("Dancing", controller.get_wiimote_buttons(0)))
        # (Optional) release after signaling
        a_release(); b_release()
        continue
