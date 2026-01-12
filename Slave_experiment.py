from dolphin import event, controller
from multiprocessing.connection import Client

PORT = 26330
AUTHKEY = b"secret password"

conn = Client(("localhost", PORT), authkey=AUTHKEY)
conn.send("READY")

# ---- Action timing ----
FRAMES_PER_ACTION = 3          # how many emulator frames each action is held
TICK_EVERY = 1                 # send a TICK every N frames (1 = every frame)

# ---- Wiimote accel mapping ----
scale = 1
base = 100
MAX_ACCEL = 32767

def _clamp_int(x: float) -> int:
    return int(max(-MAX_ACCEL, min(MAX_ACCEL, round(x))))

def _v(v: float) -> int:
    return _clamp_int(v * scale)

def set_accel(controller_id: int, x: float, y: float, z: float):
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
    1: left, 2: right, 3: up, 4: down, 5: forward, 6: backward,
    7: up_left, 8: up_right, 9: down_left, 10: down_right,
    11: forward_up, 12: forward_down, 13: forward_left, 14: forward_right,
    15: backward_up, 16: backward_down, 17: backward_left, 18: backward_right,
    19: forward_up_left, 20: forward_up_right, 21: forward_down_left, 22: forward_down_right,
    23: backward_up_left, 24: backward_up_right, 25: backward_down_left, 26: backward_down_right,
}

def a_press():   controller.set_wiimote_buttons(0, {"A": True})
def a_release(): controller.set_wiimote_buttons(0, {"A": False})
def b_press():   controller.set_wiimote_buttons(0, {"B": True})
def b_release(): controller.set_wiimote_buttons(0, {"B": False})

startup = 0

async def tick_send():
    # send current buttons as a "clock tick"
    conn.send(("TICK", controller.get_wiimote_buttons(0)))

while True:
    cmd, payload = conn.recv()

    if cmd == "send":
        # One-time startup combo if you rely on B being held to start
        if controller.get_wiimote_buttons(0)["B"] and startup < 1:
            a_press()
            b_press()
            startup += 1
            await event.frameadvance()
            await tick_send()
            conn.send(("CLOSED", controller.get_wiimote_buttons(0)))
            a_release()
            b_release()

        # Execute chosen action for FRAMES_PER_ACTION frames
        if payload in ACTIONS:
            for i in range(FRAMES_PER_ACTION):
                await event.frameadvance()
                ACTIONS[payload]()

                if (i % TICK_EVERY) == 0:
                    await tick_send()

            # Signal the end of this action window
            conn.send(("CLOSED", controller.get_wiimote_buttons(0)))
            continue

    if cmd == "reset":
        # Your existing reset macro (kept), but now emits ticks periodically
        for i in range(600):
            await event.frameadvance()
            if (i % 30) == 0:
                await tick_send()

        for i in range(3):
            conn.send(("PRINT", "a_release"))
            a_release()
            for j in range(250):
                await event.frameadvance()
                controller.set_wiimote_pointer(0, -0.7, -0.9)
                if (j % 30) == 0:
                    await tick_send()
            a_press()
            await event.frameadvance()
            await tick_send()

        a_press()
        b_press()
        await event.frameadvance()
        await tick_send()

        conn.send(("PRINT", controller.get_wiimote_buttons(0)))
        conn.send(("CLOSED", controller.get_wiimote_buttons(0)))
