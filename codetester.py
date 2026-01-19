import subprocess
from multiprocessing.connection import Listener
from collections import defaultdict
import numpy as np
from colour_grapping_2 import read_rgb, classify_pixel, reward_detector,reset_reward_state
import pickle
from pathlib import Path
import csv

# Change these paths to your own felk-dolphin version, your just dance 2 game and our slave-script
Felk_dolphin_exe = r"C:/Users/esben/Downloads/dolphin-scripting-preview4-x64/dolphin"
Game_path    = r"C:/Users/esben/Downloads/dolphin-2512-x64/Dolphin-x64/spil/Just_dance2.wbfs"
Slave_path = r"C:/Users/esben/OneDrive/Documents/GitHub/Just-Dance-Project/slavetest.py"

LOG_CSV = Path("episode_rewards_5frames.csv")
LOG_PNG = Path("episode_rewards_5frames.png")
episode_reward_total = 0.0  # running total for current episode

n_actions = 7

song_moves = 97 #233 for the long version

agent_path = "agent_for_reward_5frames.pkl"

def default_value():
    return np.zeros(n_actions, dtype=np.float32)

Q = defaultdict(default_value)
E = defaultdict(default_value)  # eligibility traces

def log_episode_reward(ep: int, total: float, path=LOG_CSV):
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["episode", "episode_reward"])
        w.writerow([ep, float(total)])

alpha = 0.05
gamma = 0.95
lam = 0.90

eps = 1.0
eps_min = 0.05
eps_decay = 0.995  # pr episode

episode_count = 0

def epsilon_greedy(state):
    global eps
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[state]))


def state_from_phase(phase):
    return phase  # state is just phase index

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



def save_agent(path=agent_path):
    data = {
        "Q": dict(Q),            # convert defaultdict -> normal dict
        "eps": eps,
        "episode_count": episode_count,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_agent(path=agent_path):
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
    global eps, episode_count
    global phase, state, a, ep_reward, episode_reward_total


    moves = 0
    total_frames = 0
    ready = False

    # start first action
    conn.send(("send", int(a)))
    while True:
        reply, payload = conn.recv()
        
        if ready:
            reward,moves,total_frames = reward_detector()
            if reward is not None:
                add_reward(reward)

        if reply == "Dancing" and moves < song_moves and (total_frames<1000 or ready == False):

            if payload['B'] and payload['A']: #Resets if A and B is pressed, which happens at the end of reset in the slave
                episode_reward_total = 0.0
                print("episode:",episode_count)
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
                if episode_count % 5 == 0 and moves == 0:
                    print("Q stats:", q_stats(Q), "eps:", eps)

            #Start of SARSA
            # reward observed during the last action window
            reward = pop_reward()
            ep_reward += reward
            episode_reward_total += reward

            # next state (current design: phase increments)
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

            #E.clear()

            # send next action to slave
            conn.send(("send", int(a)))


        else:
            if ready:
                #Resets the game
                conn.send(("reset", ":)"))
                if moves == song_moves:
                    log_episode_reward(episode_count, episode_reward_total)
                    save_agent()
                    episode_count += 1
                    try:
                        import matplotlib.pyplot as plt
                        import pandas as pd

                        df = pd.read_csv(LOG_CSV)
                        plt.figure()
                        plt.plot(df["episode"], df["episode_reward"])
                        plt.xlabel("episode_count")
                        plt.ylabel("episode_reward")
                        plt.tight_layout()
                        plt.savefig(LOG_PNG)
                        plt.close()
                    except Exception as e:
                        print("[LOG] plot skipped:", e)

                reset_reward_state()
                ready = False
                moves = 0
                total_frames = 0
                print("reset")


PORT = 26330
AUTHKEY = b"secret password"

listener = Listener(("localhost", PORT), authkey=AUTHKEY)


cmd = [Felk_dolphin_exe, "--no-python-subinterpreters", "--script", Slave_path, "-b", "--exec", Game_path]

print("[Master] launching:", cmd)
Opens_dolphin = subprocess.Popen(cmd)

print("[Master] waiting for slave connect...")
conn = listener.accept()
print("[Master] connected")

msg = conn.recv()
print("[Master] received handshake:", msg)

if __name__ == "__main__":
    dolphin_conn_loop(conn)
