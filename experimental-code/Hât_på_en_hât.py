import subprocess
from multiprocessing.connection import Listener
from collections import defaultdict
import numpy as np
from colour_grapping_2 import reward_detector, reset_reward_state
import pickle
from pathlib import Path
import csv

LOG_CSV = Path("episode_rewards.csv")
LOG_PNG = Path("episode_rewards.png")

# Change these paths
Felk_dolphin_exe = r"C:/Users/esben/Downloads/dolphin-scripting-preview4-x64/dolphin"
Game_path    = r"C:/Users/esben/Downloads/dolphin-2512-x64/Dolphin-x64/spil/Just_dance2.wbfs"
Slave_path   = r"C:/Users/esben/OneDrive/Documents/GitHub/Just-Dance-Project/slave2.py"


# ----------------------------
# Action design
# 0..6  base actions (idle + 6 one-directional)
# 0..6  add  actions (idle + 6 one-directional) applied in stage 2
# ----------------------------
N_BASE = 7
N_ADD  = 7

song_moves = 97
agent_path = "agent_2stage_14.pkl"

alpha = 0.05
gamma = 0.95
lam   = 0.90

# eps for stage 1, stage 2
eps1 = 1.0
eps2 = 0.5
eps_min = 0.05
eps_decay = 0.995

episode_count = 0

# stage schedule
STAGE1_EPISODES = 500  # train base only
# stage2 starts after that

episode_reward_total = 0.0  # running total for current episode

def log_episode_reward(ep: int, total: float, path=LOG_CSV):
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["episode", "episode_reward"])
        w.writerow([ep, float(total)])

def zeros7():
    return np.zeros(7, dtype=np.float32)

Q_base = defaultdict(zeros7)   # (7,)
E_base = defaultdict(zeros7)

Q_add  = defaultdict(zeros7)   # (7,)
E_add  = defaultdict(zeros7)

def state_from_phase(phase):
    return phase

reward_acc = 0.0

def add_reward(reward: float):
    global reward_acc
    reward_acc += float(reward)

def pop_reward() -> float:
    global reward_acc
    r = float(reward_acc)
    reward_acc = 0.0
    return r

def q_stats7(Q):
    if len(Q) == 0:
        return "empty"
    arr = np.stack([Q[s] for s in Q.keys()], axis=0)
    return {
        "num_states": int(arr.shape[0]),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "abs_mean": float(np.abs(arr).mean()),
    }

def epsilon_greedy_7(Qrow, eps):
    if np.random.rand() < eps:
        return int(np.random.randint(7))
    return int(np.argmax(Qrow))

def greedy_7(Qrow):
    return int(np.argmax(Qrow))

def save_agent(path=agent_path):
    data = {
        "Q_base": dict(Q_base),
        "Q_add": dict(Q_add),
        "eps1": eps1,
        "eps2": eps2,
        "episode_count": episode_count,
        "alpha": alpha,
        "gamma": gamma,
        "lam": lam,
        "STAGE1_EPISODES": STAGE1_EPISODES,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_agent(path=agent_path):
    global eps1, eps2, episode_count
    with open(path, "rb") as f:
        data = pickle.load(f)

    Q_base.clear()
    for k, v in data.get("Q_base", {}).items():
        Q_base[k] = np.asarray(v, dtype=np.float32)

    Q_add.clear()
    for k, v in data.get("Q_add", {}).items():
        Q_add[k] = np.asarray(v, dtype=np.float32)

    eps1 = float(data.get("eps1", eps1))
    eps2 = float(data.get("eps2", eps2))
    episode_count = int(data.get("episode_count", episode_count))

try:
    load_agent()
    print("[Master] Agent loaded")
except FileNotFoundError:
    print("[Master] No saved agent found, starting fresh")

def in_stage1():
    return episode_count < STAGE1_EPISODES

def choose_action(state, mode=None):
    """
    mode:
      None     -> pick based on in_stage1()
      "send"   -> force base-only action (int)
      "send2"  -> force (base, add) tuple
    """
    global eps1, eps2

    if mode is None:
        mode = "send" if in_stage1() else "send2"

    if mode == "send":
        a_base = epsilon_greedy_7(Q_base[state], eps1)
        return ("send", a_base)

    if mode == "send2":
        a_base = greedy_7(Q_base[state])
        a_add  = epsilon_greedy_7(Q_add[state], eps2)
        return ("send2", (a_base, a_add))

    raise ValueError(mode)

def update_sarsa_lambda(cur_cmd, state, action_payload, reward, s2, next_action_payload):
    """
    If cur_cmd == "send":   learn Q_base/E_base on an int action (0..6)
    If cur_cmd == "send2":  learn Q_add/E_add on the add action (payload[1])
    """
    if cur_cmd == "send":
        a  = int(action_payload)
        a2 = int(next_action_payload)

        delta = reward + gamma * Q_base[s2][a2] - Q_base[state][a]
        E_base[state][a] += 1.0

        for s_key in list(E_base.keys()):
            Q_base[s_key] += alpha * delta * E_base[s_key]
            E_base[s_key] *= gamma * lam
            if np.max(np.abs(E_base[s_key])) < 1e-6:
                del E_base[s_key]
        return

    if cur_cmd == "send2":
        # payload is (base_id, add_id) but we learn only add_id
        a_add  = int(action_payload[1])
        a2_add = int(next_action_payload[1])

        delta = reward + gamma * Q_add[s2][a2_add] - Q_add[state][a_add]
        E_add[state][a_add] += 1.0

        for s_key in list(E_add.keys()):
            Q_add[s_key] += alpha * delta * E_add[s_key]
            E_add[s_key] *= gamma * lam
            if np.max(np.abs(E_add[s_key])) < 1e-6:
                del E_add[s_key]
        return

    raise ValueError(f"Unknown cmd: {cur_cmd}")

def dolphin_conn_loop(conn):
    global eps1, eps2, episode_count, episode_reward_total

    moves = 0
    total_frames = 0
    ready = False

    phase = 0
    state = state_from_phase(phase)

    # clear traces at start
    E_base.clear()
    E_add.clear()

    cur_cmd, action_payload = choose_action(state)
    conn.send((cur_cmd, action_payload))

    while True:
        reply, payload = conn.recv()

        if ready:
            reward, moves, total_frames = reward_detector()
            if reward is not None:
                add_reward(reward)

        if reply == "Dancing" and moves < song_moves and (total_frames < 1300 or ready is False):

            # episode start marker: slave presses A+B at end of reset
            if payload.get("B", False) and payload.get("A", False):
                episode_reward_total = 0.0
                print("episode:", episode_count, "stage:", 1 if in_stage1() else 2)
                moves = 0
                phase = 0
                total_frames = 0
                ready = True
                state = state_from_phase(phase)

                E_base.clear()
                E_add.clear()

                # decay epsilon per episode
                if in_stage1():
                    eps1 = max(eps_min, eps1 * eps_decay)
                else:
                    eps2 = max(eps_min, eps2 * eps_decay)

                _ = pop_reward()

                if episode_count % 5 == 0:
                    print("Q_base:", q_stats7(Q_base), "eps1:", eps1)
                    if not in_stage1():
                        print("Q_add :", q_stats7(Q_add),  "eps2:", eps2)
                cur_cmd, action_payload = choose_action(state)
                conn.send((cur_cmd, action_payload))
                continue
            # SARSA step
            reward = pop_reward()
            episode_reward_total += reward

            phase += 1
            s2 = state_from_phase(phase)

            next_cmd, next_action_payload = choose_action(s2, mode=cur_cmd)
            update_sarsa_lambda(cur_cmd, state, action_payload, reward, s2, next_action_payload)

            state = s2
            cur_cmd = next_cmd
            action_payload = next_action_payload

            conn.send((cur_cmd, action_payload))

        else:
            if ready:
                conn.send(("reset", ":)"))
                if moves == song_moves:
                    # log reward BEFORE incrementing episode_count
                    log_episode_reward(episode_count, episode_reward_total)

                    save_agent()

                    # optional: also update a png plot
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

                    episode_count += 1
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
