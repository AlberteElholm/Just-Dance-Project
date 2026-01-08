import subprocess
import platform
from multiprocessing.connection import Listener
import random
import time
import mss
from pynput import mouse
import threading

m = mouse.Controller()

count = 0

PIXEL_X = 280
PIXEL_Y = 446

POLL_HZ = 248
ARM_ON_UNKNOWN = False
DEBUG = False

BPM = 124
SPEED_MULTIPLIER = 10
COOLDOWN_MS = 500 / SPEED_MULTIPLIER

sct = mss.mss()

def read_rgb(x, y):
    img = sct.grab({"top": y, "left": x, "width": 1, "height": 1})
    r, g, b = img.pixel(0, 0)
    return r, g, b

def classify_pixel(r, g, b):
    if r < 45 and g < 50 and b < 50:
        return "None"
    if r > 120 and g < 60 and b < 60:
        return "X"
    if r > 200 and g > 150 and b < 80:
        return "Yeah"
    if g > 170 and r > 100:
        return "Perfect"
    if b > 200 and r < 20:
        return "Good"
    if r > 120 and b > 120:
        return "OK"
    return "Unknown"

def detect_point_events():
    dt = 1.0 / POLL_HZ
    armed = True
    last_event_time = 0.0
    last_t = None

    while True:
        r, g, b = read_rgb(PIXEL_X, PIXEL_Y)
        label = classify_pixel(r, g, b)

        now = time.time()
        cooldown_ok = (now - last_event_time) * 1000.0 >= COOLDOWN_MS

        is_judgement = label in {"X", "OK", "Good", "Perfect", "Yeah"}
        is_clear = (label == "None") or (ARM_ON_UNKNOWN and label == "Unknown")

        if is_clear:
            armed = True

        if armed and is_judgement and cooldown_ok:
            event_dt = 0.0 if last_t is None else (now - last_t)
            last_t = now
            last_event_time = now
            armed = False

            yield {"t": now, "dt": event_dt, "label": label, "rgb": (r, g, b)}

        if DEBUG:
            print(f"RGB={r,g,b}  label={label}  armed={armed}")

        time.sleep(dt)

# --- NEW: run conn loop in a thread ---
def dolphin_conn_loop(conn):
    # send an initial move (optional)
    move = random.randint(0, 26)
    conn.send(("send", move))

    while True:
        reply, payload = conn.recv()
        if reply == "CLOSED":
            move = random.randint(0, 26)
            conn.send(("send", move))

PORT = 26330
AUTHKEY = b"secret password"
DOLPHIN_EXE = r"C:/Users/esben/Downloads/dolphin-scripting-preview4-x64/dolphin"
ISO_PATH    = r"C:/Users/esben/Downloads/dolphin-2512-x64/Dolphin-x64/spil/Just_dance2.wbfs"
SCRIPT_PATH = r"C:/Users/esben/OneDrive/Documents/GitHub/Just-Dance-Project/slavetest.py"

listener = Listener(("localhost", PORT), authkey=AUTHKEY)

sysname = platform.system()
if sysname == "Windows":
    cmd = [DOLPHIN_EXE, "--no-python-subinterpreters", "--script", SCRIPT_PATH, "-b", "--exec", ISO_PATH]
elif sysname == "Linux":
    cmd = [DOLPHIN_EXE, "--no-python-subinterpreters", "--script", SCRIPT_PATH, "-b", f"--exec={ISO_PATH}"]
elif sysname == "Darwin":
    cmd = ["open", DOLPHIN_EXE, "--args", "--no-python-subinterpreters", "--script", SCRIPT_PATH, "-b", f"--exec={ISO_PATH}"]
else:
    raise RuntimeError("Unsupported OS")

print("[Master] launching:", cmd)
proc = subprocess.Popen(cmd)

print("[Master] waiting for slave connect...")
conn = listener.accept()
print("[Master] connected")

msg = conn.recv()
print("[Master] received handshake:", msg)

if __name__ == "__main__":
    # start Dolphin connection loop concurrently
    t = threading.Thread(target=dolphin_conn_loop, args=(conn,), daemon=True)
    t.start()

    print("Listening for new point-message events... (Ctrl+C to stop)")
    for ev in detect_point_events():
        print(f"EVENT: label={ev['label']} dt={ev['dt']:.3f}s rgb={ev['rgb']} count: {count}")
        count += 1
