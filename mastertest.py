import subprocess
import platform
from multiprocessing.connection import Listener
import random

PORT = 26330
AUTHKEY = b"secret password"
DOLPHIN_EXE = r"C:/Users/esben/Downloads/dolphin-scripting-preview4-x64/dolphin"         # change this
ISO_PATH    = r"C:/Users/esben/Downloads/dolphin-2512-x64/Dolphin-x64/spil/Just_dance2.wbfs"  # change this
SCRIPT_PATH = r"C:/Users/esben/OneDrive/Documents/GitHub/Just-Dance-Project/slavetest.py"  # change this

listener = Listener(("localhost", PORT), authkey=AUTHKEY)

# Launch Dolphin with script
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

# Handshake
msg = conn.recv()
print("[Master] received handshake:", msg)

move = random.randint(0,26)
conn.send(("send",move))
while True:
    reply, payload = conn.recv()
    if reply=="CLOSED":
        move = random.randint(0,26)
        conn.send(("send",move))
        print("[Master] advance reply:", payload)
