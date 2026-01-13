import subprocess
import platform
from multiprocessing.connection import Listener
import mss
from collections import defaultdict
import numpy as np
from colour_grapping_2 import read_rgb, classify_pixel
import pickle
from pathlib import Path


def default_value():
    return np.zeros(27, dtype=np.float32)

Q = defaultdict(default_value)
E = defaultdict(default_value)  # eligibility traces



alpha = 0.05
gamma = 0.95
lam = 0.90

eps = 1.0
eps_min = 0.05
eps_decay = 0.995  # pr episode

def epsilon_greedy(state):
    if np.random.rand() < eps:
        return np.random.randint(27)
    return int(np.argmax(Q[state]))

def state_from_phase(phase):
    return phase  # state is just phase index

episode_count = 0
move_count = 0 # antal moves talt

PIXEL_X = 287 #280 rasputin  
PIXEL_Y = 422 #553  

ARM_ON_UNKNOWN = False
DEBUG = False

cd_frames = 10
song_moves = 97 #233

phase = 0
state = state_from_phase(phase)

# clear traces at start of episode
E.clear()

a = epsilon_greedy(state)

ep_reward = 0.0

#reward_lock = threading.Lock()
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
    #with reward_lock:
    reward_acc += reward

def pop_reward() -> float:
    global reward_acc
    #with reward_lock:
    reward = reward_acc
    reward_acc = 0.0
    return reward

def save_agent(path="agent.pkl"):
    data = {
        "Q": dict(Q),            # convert defaultdict -> normal dict
        "eps": eps,
        "episode_count": episode_count,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_agent(path="agent.pkl"):
    global eps, episode_count
    with open(path, "rb") as f:
        data = pickle.load(f)
    Q.clear()
    for k, v in data["Q"].items():
        Q[k] = v
    eps = data.get("eps", eps)
    episode_count = data.get("episode_count", episode_count)

try:
    load_agent()
    print("[Master] Agent loaded")
except FileNotFoundError:
    print("[Master] No saved agent found, starting fresh")

# --- NEW: run conn loop in a thread ---
def dolphin_conn_loop(conn):
    global eps, episode_count, move_count
    global phase, state, a, ep_reward

    armed = True
    frame = 0
    moves = 0
    total_frames = 0
    prev_label = 0
    ready = False
    # start first action
    conn.send(("send", int(a)))
    sct = mss.mss()
    while True:
        reply, payload = conn.recv()
        
        if ready:
            frame += 1
            total_frames += 1
            r, g, b = read_rgb(PIXEL_X, PIXEL_Y)
            label = classify_pixel(r, g, b)

            cooldown_ok = frame >= cd_frames

            is_judgement = label[0] in {"X", "OK", "Good", "Perfect", "Yeah"}

            label_changed = prev_label != label[0]

            is_clear = (label[0] == "None") or (ARM_ON_UNKNOWN and label[0] == "Unknown") or label_changed

            prev_label = label[0]
            if is_clear:
                armed = True

            # Fire event on first appearance
            if armed and is_judgement and cooldown_ok:
                moves += 1
                event_dt = frame
                frame = 0
                armed = False
                reward = label[1]
                print("rgb:",r,g,b,label,"frames:",event_dt,"moves:", moves, "total_frames:",total_frames)
                add_reward(float(label[1]))

        if reply == "CLOSED" and moves < song_moves and (total_frames<2000 or ready == False):

            if payload['B'] and payload['A']: #automatisk reset 

                print("episode:",episode_count)
                move_count = 0
                moves = 0
                phase = 0
                total_frames = 0
                ready = True
                state = state_from_phase(phase)

                E.clear()
                a = epsilon_greedy(state)

                ep_reward = 0.0

                # decay epsilon per episode (recommended)
                eps = max(eps_min, eps * eps_decay)

                # clear any leftover reward
                _ = pop_reward()
                if episode_count % 5 == 0 and move_count == 0:
                    print("Q stats:", q_stats(Q), "eps:", eps)

            # reward observed during the last action window
            reward = pop_reward()
            ep_reward += reward
            move_count += 1

            # next state (your current design: phase increments)
            phase += 1
            s2 = state_from_phase(phase)

            # choose next action
            a2 = epsilon_greedy(s2)

            # SARSA(λ) TD error
            delta = reward + gamma * Q[s2][a2] - Q[state][a]

            # eligibility trace update (accumulating traces)
            E[state][a] += 1.0

            # update all traced pairs
            for s_key in list(E.keys()):
                Q[s_key] += alpha * delta * E[s_key]
                E[s_key] *= gamma * lam
                if np.max(np.abs(E[s_key])) < 1e-6:
                    del E[s_key]

            state, a = s2, a2

            # send next action to slave
            conn.send(("send", int(a)))

            #print("ep", episode_count, "move", move_count, "R", ep_reward, "eps", eps, payload)
        elif reply == "waiting":
            conn.send(("filler", ":)"))
            print(payload)
        
        elif reply == "print":
            conn.send(("filler", ":)"))
            print(payload)

        else:
            if ready:
                conn.send(("reset", ":)"))
                if moves == song_moves:
                    save_agent()
                    episode_count += 1
                ready = False
                moves = 0
                total_frames = 0
                print("reset",reply,payload)


PORT = 26330
AUTHKEY = b"secret password"
DOLPHIN_EXE = r"C:/Users/esben/Downloads/dolphin-scripting-preview4-x64/dolphin"
ISO_PATH    = r"C:/Users/esben/Downloads/dolphin-2512-x64/Dolphin-x64/spil/Just_dance2.wbfs"
SCRIPT_PATH = r"C:/Users/esben/OneDrive/Documents/GitHub/Just-Dance-Project/slavetest.py"

listener = Listener(("localhost", PORT), authkey=AUTHKEY)

sysname = platform.system()
cmd = [DOLPHIN_EXE, "--no-python-subinterpreters", "--script", SCRIPT_PATH, "-b", "--exec", ISO_PATH]

print("[Master] launching:", cmd)
Opens_dolphin = subprocess.Popen(cmd)

print("[Master] waiting for slave connect...")
conn = listener.accept()
print("[Master] connected")

msg = conn.recv()
print("[Master] received handshake:", msg)

if __name__ == "__main__":
    dolphin_conn_loop(conn)
