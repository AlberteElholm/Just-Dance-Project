# q_table_heatmap_plus_actions_first1500_2stage14.py
# Copy–paste and run

import pickle
import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
PKL_PATH = "agent_2stage_14.pkl"
MAX_STATES = 1250
MODE = "both"   # "base", "add", or "both"
# ==================

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

Qb = data.get("Q_base", {})
Qa = data.get("Q_add", {})

if MODE not in {"base", "add", "both"}:
    raise ValueError("MODE must be 'base', 'add', or 'both'")

# collect states from whichever tables exist
state_set = set()
state_set.update(Qb.keys())
state_set.update(Qa.keys())
all_states = sorted(state_set)

if len(all_states) == 0:
    raise ValueError("Both Q_base and Q_add are empty")

states = all_states[:MAX_STATES]

def get_row(Qdict, s, width=7):
    v = Qdict.get(s, None)
    if v is None:
        return np.zeros(width, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if v.shape[0] != width:
        # pad/truncate defensively
        out = np.zeros(width, dtype=np.float32)
        m = min(width, v.shape[0])
        out[:m] = v[:m]
        return out
    return v

if MODE == "base":
    Q = np.stack([get_row(Qb, s, 7) for s in states], axis=0)
    action_names = [str(i) for i in range(7)]
    title = f"Q_base heatmap (first {len(states)} states)"

elif MODE == "add":
    Q = np.stack([get_row(Qa, s, 7) for s in states], axis=0)
    action_names = [str(i) for i in range(7)]
    title = f"Q_add heatmap (first {len(states)} states)"

else:  # both
    Q = np.stack([np.concatenate([get_row(Qb, s, 7), get_row(Qa, s, 7)]) for s in states], axis=0)
    action_names = [str(i) for i in range(14)]
    title = f"Q_base|Q_add concatenated heatmap (first {len(states)} states)"

n_states, n_actions = Q.shape

print("Loaded 2-stage Q-tables")
print(f"Mode: {MODE}")
print(f"Visualizing states: {n_states} / {len(all_states)} (MAX_STATES={MAX_STATES})")
print("Actions:", n_actions)
print("Q min / max:", float(Q.min()), float(Q.max()))

plt.figure(figsize=(12, 6))
plt.imshow(Q, aspect="auto", interpolation="none")
plt.colorbar(label="Q-value")
plt.xlabel("Action index")
plt.ylabel("State (phase)")
plt.title(title)
plt.tight_layout()
plt.show()
