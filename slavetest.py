from dolphin import event, controller
from multiprocessing.connection import Client
PORT = 26330
AUTHKEY = b"secret password"

conn = Client(("localhost", PORT), authkey=AUTHKEY)
conn.send("READY")

frames_per_action = 4

base = 100  # base acceleration

def set_accel(controller_id:int,x: float, y: float, z: float):
    controller.set_wiimote_acceleration(controller_id, x, y, z)

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

    # 1D
    1: left,  2: right,  3: up,    4: down,  5: forward,  6: backward,

    # 2D
    7: up_left,    8: up_right,    9: down_left,    10: down_right,
    11: forward_up, 12: forward_down, 13: forward_left, 14: forward_right,
    15: backward_up, 16: backward_down, 17: backward_left, 18: backward_right,

    # 3D
    #19: forward_up_left,   20: forward_up_right,
    #21: forward_down_left, 22: forward_down_right,
    #23: backward_up_left,  24: backward_up_right,
    #25: backward_down_left,26: backward_down_right,
}

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
    controller.set_wiimote_pointer(0,-0.7,-0.9)

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


startup = 0

episode_count = 0

while True:
    cmd, payload = conn.recv()

    if cmd == "send":
        if controller.get_wiimote_buttons(0)['B'] and startup < 1:
            a_press()
            b_press()
            startup += 1
            await event.frameadvance()
            conn.send(("Dancing",controller.get_wiimote_buttons(0)))
            a_release()
            b_release()

        if payload in ACTIONS:
            for i in range(frames_per_action):
                await event.frameadvance()
                ACTIONS[payload]()
            conn.send(("Dancing",controller.get_wiimote_buttons(0)))
            continue
    
    if cmd == "reset":
        await wait_frames(600)

        if episode_count < 2:
            episode_count += 1
            await a_sequence(hold_frames=200, repeats=2)
        else:
            await a_sequence(hold_frames=20, repeats=1)

        await wait_frames(100)

        a_press(); b_press()
        await event.frameadvance()
        a_press(); b_press()
        conn.send(("Dancing", controller.get_wiimote_buttons(0)))