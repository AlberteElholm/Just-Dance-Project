import subprocess
from multiprocessing.connection import Listener
from collections import defaultdict
import numpy as np
from colour_grapping_2 import read_rgb, classify_pixel, reward_detector, reset_reward_state
import pickle

# Change these paths to your own felk-dolphin version, your just dance 2 game and our slave-script
Felk_dolphin_exe = r"C:/Users/esben/Downloads/dolphin-scripting-preview4-x64/dolphin"
Game_path        = r"C:/Users/esben/Downloads/dolphin-2512-x64/Dolphin-x64/spil/Just_dance2.wbfs"
Slave_path       = r"C:/Users/esben/OneDrive/Documents/GitHub/Just-Dance-Project/slavetest.py"

AGENT_PATH = "agent.pkl"   # <-- opens this file

N_ACTIONS = 7

def default_value():
    return np.zeros(N_ACTIONS, dtype=np.float32)

Q = defaultdict(default_value)
E = defaultdict(default_value)  # eligibility traces (kept, but not used when greedy-only)

alpha = 0.05
gamma = 0.95
lam = 0.90

# NOTE: Greedy run (no exploration)
def greedy(state: int) -> int:
    return int(np.argmax(Q[state]))

def state_from_phase(phase):
    return phase  # state is just phase index

episode_count = 0

song_moves = 97  # 233 for the long version

phase = 0
state = state_from_phase(phase)

E.clear()

a = greedy(state)

ep_reward = 0.0
reward_acc = 0.0

def q_stats(Q):
    if len(Q) == 0:
        return "Q is empty"
    arr = np.stack([Q[s] for s in Q.keys()], axis=0)  # (num_states, 27)
    return {
        "num_states": arr.shape[0],
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "abs_mean": float(np.abs(arr).mean()),
    }

def add_reward(reward: float):
    global reward_acc
    reward_acc += reward

def pop_reward() -> float:
    global reward_acc
    reward = reward_acc
    reward_acc = 0.0
    return reward

#def save_agent(path="agent.pkl"):
#    data = {
#        "Q": dict(Q),
#        "episode_count": episode_count,
#    }
#    with open(path, "wb") as f:
#        pickle.dump(data, f)

def load_agent(path=AGENT_PATH):
    global episode_count
    with open(path, "rb") as f:
        data = pickle.load(f)

    Q.clear()
    for k, v in data["Q"].items():
        # ensure numpy arrays (safety if pickle contains lists)
        Q[k] = np.array(v, dtype=np.float32)

    episode_count = data.get("episode_count", episode_count)

try:
    load_agent()
    print(f"[Master] Agent loaded from: {AGENT_PATH}")
    print("[Master] Q stats:", q_stats(Q))
except FileNotFoundError:
    print(f"[Master] Could not find {AGENT_PATH}. Put it next to this script or change AGENT_PATH.")
    raise

# --- run conn loop ---
def dolphin_conn_loop(conn):
    global phase, state, a, ep_reward, episode_count

    moves = 0
    total_frames = 0
    ready = False

    # start first action
    conn.send(("send", int(a)))

    while True:
        reply, payload = conn.recv()

        if ready:
            reward, moves, total_frames = reward_detector()
            if reward is not None:
                add_reward(reward)

        if reply == "Dancing" and moves < song_moves and (total_frames < 2000 or ready is False):

            if payload.get("B") and payload.get("A"):  # automatic reset / new run
                print("episode:", episode_count)
                moves = 0
                phase = 0
                total_frames = 0
                ready = True

                state = state_from_phase(phase)
                a = greedy(state)     # <-- greedy action

                ep_reward = 0.0
                _ = pop_reward()

            # reward observed during the last action window
            reward = pop_reward()
            ep_reward += reward

            # next state
            phase += 1
            s2 = state_from_phase(phase)

            # next action (greedy)
            a2 = greedy(s2)          # <-- greedy action

            # (No learning updates in greedy evaluation mode)
            state, a = s2, a2

            # send next action to slave
            conn.send(("send", int(a)))

        elif reply == "waiting":
            conn.send(("filler", ":)"))
            print(payload)

        elif reply == "print":
            conn.send(("filler", ":)"))
            print(payload)

        else:
            if ready:
                conn.send(("reset", ":)"))
                reset_reward_state()
                ready = False
                moves = 0
                total_frames = 0
                episode_count += 1
                print("reset (greedy run finished). total ep_reward:", ep_reward)

PORT = 26330
AUTHKEY = b"secret password"

listener = Listener(("localhost", PORT), authkey=AUTHKEY)

cmd = [
    Felk_dolphin_exe,
    "--no-python-subinterpreters",
    "--script",
    Slave_path,
    "-b",
    "--exec",
    Game_path,
]

print("[Master] launching:", cmd)
Opens_dolphin = subprocess.Popen(cmd)

print("[Master] waiting for slave connect...")
conn = listener.accept()
print("[Master] connected")

msg = conn.recv()
print("[Master] received handshake:", msg)

if __name__ == "__main__":
    dolphin_conn_loop(conn)
