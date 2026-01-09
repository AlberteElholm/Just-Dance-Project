import subprocess
import platform
from multiprocessing.connection import Listener
import random
import time
import mss
from pynput import mouse
import threading
from collections import defaultdict
import numpy as np
from colour_grapping_2 import *
from queue import Queue, Empty
import pickle
from pathlib import Path


m = mouse.Controller()

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

PIXEL_X = 280
PIXEL_Y = 446

ARM_ON_UNKNOWN = False
DEBUG = False

#BPM = 124
#POLL_HZ = 2*BPM
SPEED_MULTIPLIER = 10
cd_frames = 20

phase = 0
state = state_from_phase(phase)

# clear traces at start of episode
E.clear()

a = epsilon_greedy(state)

ep_reward = 0.0

reward_lock = threading.Lock()
reward_acc = 0.0



def add_reward(reward: float):
    global reward_acc
    with reward_lock:
        reward_acc += reward

def pop_reward() -> float:
    global reward_acc
    with reward_lock:
        reward = reward_acc
        reward_acc = 0.0
        return reward

# --- NEW: run conn loop in a thread ---
def dolphin_conn_loop(conn):
    global eps, episode_count, move_count
    global phase, state, a, ep_reward

    armed = True
    frame = 0
    moves = 0
    # start first action
    conn.send(("send", int(a)))
    sct = mss.mss()
    while True:
        reply, payload = conn.recv()
        frame += 1
        

        r, g, b = read_rgb(PIXEL_X, PIXEL_Y)
        label = classify_pixel(r, g, b)

        cooldown_ok = frame >= cd_frames

        is_judgement = label[0] in {"X", "OK", "Good", "Perfect", "Yeah"}
        is_clear = (label[0] == "None") or (ARM_ON_UNKNOWN and label[0] == "Unknown")

        if is_clear:
            armed = True

        # Fire event on first appearance
        if armed and is_judgement and cooldown_ok:
            moves += 1
            event_dt = frame
            frame = 0
            armed = False
            reward = label[1]
            print(label,"frames:",event_dt,"moves:", move_count)
            add_reward(float(label[1]))

        if reply == "CLOSED":
            
            if payload['B'] and not payload['A']: # manuel reset
                
                move_count = 0

                phase = 0
                state = state_from_phase(phase)

                E.clear()
                a = epsilon_greedy(state)

                ep_reward = 0.0

                # decay epsilon per episode (recommended)
                eps = max(eps_min, eps * eps_decay)

                # clear any leftover reward
                _ = pop_reward()

                conn.send(("send", int(a)))
                continue
            
            if payload['B'] and payload['A']: #automatisk reset 
                episode_count += 1

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
            delta = r + gamma * Q[s2][a2] - Q[state][a]

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


PORT = 26330
AUTHKEY = b"secret password"
DOLPHIN_EXE = r"C:/Users/esben/Downloads/dolphin-scripting-preview4-x64/dolphin"
ISO_PATH    = r"C:/Users/esben/Downloads/dolphin-2512-x64/Dolphin-x64/spil/Just_dance2.wbfs"
SCRIPT_PATH = r"C:/Users/esben/OneDrive/Documents/GitHub/Just-Dance-Project/slavetest.py"

listener = Listener(("localhost", PORT), authkey=AUTHKEY)

sysname = platform.system()
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
    # start Dolphin connection loop concurrently
#    t = threading.Thread(target=dolphin_conn_loop, args=(conn,), daemon=True)
#    t.start()

#    print("Listening for new point-message events... (Ctrl+C to stop)")
#    for ev in detect_point_events():
#        label, reward = ev["label"]   # because label is ["Perfect", 1.0]
#        add_reward(float(reward))
