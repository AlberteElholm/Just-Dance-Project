from dolphin import event, controller
from multiprocessing.connection import Client
PORT = 26330
AUTHKEY = b"secret password"

conn = Client(("localhost", PORT), authkey=AUTHKEY)
conn.send("READY")

count=0

#actions
frames_per_action = 3

scale = 1          # scalar applied to all actions
base = 100           # base accel magnitude before scaling
MAX_ACCEL = 32767    # typical 16-bit safe clamp


def _clamp_int(x: float) -> int:
    return int(max(-MAX_ACCEL, min(MAX_ACCEL, round(x))))

def _v(v: float) -> int:
    return _clamp_int(v * scale)

def set_accel(controller_id:int,x: float, y: float, z: float):
    controller.set_wiimote_acceleration(controller_id, _v(x), _v(y), _v(z))

def idle():          set_accel(0, 0, 0, 0)

def left():          set_accel(0, -base, 0, 0)
def right():         set_accel(0,  base, 0, 0)
def up():            set_accel(0, 0,  base, 0)
def down():          set_accel(0, 0, -base, 0)
def forward():       set_accel(0, 0, 0,  base)
def backward():      set_accel(0, 0, 0, -base)

def up_left():       set_accel(0, -base,  base, 0)
def up_right():      set_accel(0,  base,  base, 0)
def down_left():     set_accel(0, -base, -base, 0)
def down_right():    set_accel(0,  base, -base, 0)

def forward_up():    set_accel(0, 0,  base,  base)
def forward_down():  set_accel(0, 0, -base,  base)
def forward_left():  set_accel(0, -base, 0,  base)
def forward_right(): set_accel(0,  base, 0,  base)

def backward_up():   set_accel(0, 0,  base, -base)
def backward_down(): set_accel(0, 0, -base, -base)
def backward_left(): set_accel(0, -base, 0, -base)
def backward_right():set_accel(0,  base, 0, -base)

def forward_up_left():    set_accel(0, -base,  base,  base)
def forward_up_right():   set_accel(0,  base,  base,  base)
def forward_down_left():  set_accel(0, -base, -base,  base)
def forward_down_right(): set_accel(0,  base, -base,  base)

def backward_up_left():   set_accel(0, -base,  base, -base)
def backward_up_right():  set_accel(0,  base,  base, -base)
def backward_down_left(): set_accel(0, -base, -base, -base)
def backward_down_right():set_accel(0,  base, -base, -base)

ACTIONS = {
    0: idle,

    # single-axis (1D)
    1: left,
    2: right,
    3: up,
    4: down,
    5: forward,
    6: backward,

    # two-axis (2D)
    7: up_left,
    8: up_right,
    9: down_left,
    10: down_right,

    11: forward_up,
    12: forward_down,
    13: forward_left,
    14: forward_right,

    15: backward_up,
    16: backward_down,
    17: backward_left,
    18: backward_right,

    # three-axis (3D)
    19: forward_up_left,
    20: forward_up_right,
    21: forward_down_left,
    22: forward_down_right,

    23: backward_up_left,
    24: backward_up_right,
    25: backward_down_left,
    26: backward_down_right,
}

def a_press():
    controller.set_wiimote_buttons(0, {"A": True})

def a_release():
    controller.set_wiimote_buttons(0, {"A": False})

def b_press():
    controller.set_wiimote_buttons(0, {"B": True})

def b_release():
    controller.set_wiimote_buttons(0, {"B": False})


startup = 0

while True:
    cmd, payload = conn.recv()

    if cmd == "send":
        if controller.get_wiimote_buttons(0)['B'] and startup < 1:
            a_press()
            b_press()
            startup += 1
            await event.frameadvance()
            conn.send(("CLOSED",controller.get_wiimote_buttons(0)))
            a_release()
            b_release()

        if payload in ACTIONS:
            for i in range(frames_per_action):
                await event.frameadvance()
                ACTIONS[payload]()
            conn.send(("CLOSED",controller.get_wiimote_buttons(0)))
            continue
    
    if cmd == "reset":
        for i in range(400):
            await event.frameadvance()
        for i in range(3):
            a_press()
            for j in range(200):
                await event.frameadvance()
                a_release()
                controller.set_wiimote_pointer(0,-0.7, -0.9)
        await event.frameadvance()
        a_press()
        b_press()
        conn.send(("CLOSED",controller.get_wiimote_buttons(0)))
        a_release()
        b_release()