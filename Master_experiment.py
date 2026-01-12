import subprocess
import platform
from multiprocessing.connection import Listener
import threading
from collections import defaultdict
import numpy as np
import mss

from colour_grapping_2 import read_rgb, classify_pixel

# -------------------------
# RL setup
# -------------------------
N_ACTIONS = 27

def default_value():
    return np.zeros(N_ACTIONS, dtype=np.float32)

Q = defaultdict(default_value)
E = defaultdict(default_value)  # eligibility traces

# Tuned for long episodes like your song
alpha = 0.02
gamma = 0.99
lam = 0.95

eps = 1.0
eps_min = 0.01
eps_decay = 0.9995  # slower decay than before

def epsilon_greedy(state: int) -> int:
    global eps
    if np.random.rand() < eps:
        return int(np.random.randint(N_ACTIONS))
    return int(np.argmax(Q[state]))

def state_from_phase(phase: int) -> int:
    return phase

# -------------------------
# Game/pixel settings
# -------------------------
PIXEL_X = 280
PIXEL_Y = 446

ARM_ON_UNKNOWN = False
cd_frames = 10        # cooldown measured in TICKs (frames)
song_moves = 233

# -------------------------
# Reward accumulator (from judgement detection)
# -------------------------
reward_lock = threading.Lock()
reward_acc = 0.0

def add_reward(r: float):
    global reward_acc
    with reward_lock:
        reward_acc += float(r)

def pop_reward() -> float:
    global reward_acc
    with reward_lock:
        r = reward_acc
        reward_acc = 0.0
        return float(r)

# -------------------------
# Core loop
# -------------------------
def dolphin_conn_loop(conn):
    global eps

    phase = 0
    state = state_from_phase(phase)
    E.clear()

    a = epsilon_greedy(state)

    armed = True
    tick_count = 0
    moves = 0
    episode_count = 0
    ep_reward = 0.0

    # Start first action
    conn.send(("send", int(a)))

    while True:
        reply, payload = conn.recv()

        # ------------------------------------------------------------
        # Per-frame tick: read pixel and detect judgement transitions
        # ------------------------------------------------------------
        if reply == "TICK":
            tick_count += 1

            r_, g_, b_ = read_rgb(PIXEL_X, PIXEL_Y)
            label = classify_pixel(r_, g_, b_)  # expected ["Perfect", 1.0] etc.

            cooldown_ok = tick_count >= cd_frames
            is_judgement = label[0] in {"X", "OK", "Good", "Perfect", "Yeah"}
            is_clear = (label[0] == "None") or (ARM_ON_UNKNOWN and label[0] == "Unknown")

            if is_clear:
                armed = True

            if armed and is_judgement and cooldown_ok:
                armed = False
                tick_count = 0
                moves += 1
                add_reward(float(label[1]))
                print(label, "moves:", moves)

            continue

        # ------------------------------------------------------------
        # End of an action window: do ONE RL update and send next action
        # ------------------------------------------------------------
        if reply == "CLOSED":
            buttons = payload

            # Episode reset signal (your existing condition)
            # When you detect reset, clear traces & phase, decay eps, etc.
            if buttons.get("B") and buttons.get("A"):
                episode_count += 1
                print("episode:", episode_count)

                phase = 0
                state = state_from_phase(phase)
                E.clear()
                ep_reward = 0.0
                moves = 0
                tick_count = 0
                armed = True

                eps = max(eps_min, eps * eps_decay)
                _ = pop_reward()

                a = epsilon_greedy(state)
                conn.send(("send", int(a)))
                continue

            # If song is done (or you want to terminate), trigger reset
            if moves >= song_moves:
                conn.send(("reset", ":)"))
                moves = 0
                tick_count = 0
                armed = True
                continue

            # ---- This is the reward for the previous action window ----
            reward = pop_reward()
            ep_reward += reward

            # next state
            phase += 1
            s2 = state_from_phase(phase)

            # choose next action (on-policy SARSA)
            a2 = epsilon_greedy(s2)

            # -------------------------
            # SARSA(λ) update
            # IMPORTANT: use reward (not pixel red channel)
            # -------------------------
            target = reward + gamma * Q[s2][a2]
            delta = target - Q[state][a]

            # REPLACING traces (more stable)
            E[state] *= 0.0
            E[state][a] = 1.0

            for s_key in list(E.keys()):
                Q[s_key] += alpha * delta * E[s_key]
                E[s_key] *= gamma * lam
                if np.max(np.abs(E[s_key])) < 1e-8:
                    del E[s_key]

            state, a = s2, a2

            # send next action
            conn.send(("send", int(a)))
            continue

        # ------------------------------------------------------------
        # Debug / prints from slave
        # ------------------------------------------------------------
        if reply == "PRINT":
            print("[Slave]", payload)
            continue

        # ------------------------------------------------------------
        # Unknown message: force reset
        # ------------------------------------------------------------
        print("Unknown reply:", reply, payload)
        conn.send(("reset", ":)"))


# -------------------------
# Launch Dolphin + connect
# -------------------------
PORT = 26330
AUTHKEY = b"secret password"
DOLPHIN_EXE = r"C:/Users/esben/Downloads/dolphin-scripting-preview4-x64/dolphin"
ISO_PATH    = r"C:/Users/esben/Downloads/dolphin-2512-x64/Dolphin-x64/spil/Just_dance2.wbfs"
SCRIPT_PATH = r"C:/Users/esben/OneDrive/Documents/GitHub/Just-Dance-Project/slavetest.py"

listener = Listener(("localhost", PORT), authkey=AUTHKEY)

cmd = [DOLPHIN_EXE, "--no-python-subinterpreters", "--script", SCRIPT_PATH, "-b", "--exec", ISO_PATH]
print("[Master] launching:", cmd)
proc = subprocess.Popen(cmd)

print("[Master] waiting for slave connect...")
conn = listener.accept()
print("[Master] connected")

msg = conn.recv()
print("[Master] received handshake:", msg)

if __name__ == "__main__":
    dolphin_conn_loop(conn)
