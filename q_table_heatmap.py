# q_table_heatmap.py
# Copy–paste and run

import pickle
import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
PKL_PATH = "agent.pkl"   # change if needed
# ==================

# ---- Load agent ----
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

Q_dict = data["Q"]

if len(Q_dict) == 0:
    raise ValueError("Q-table is empty")

# ---- Build Q-matrix (states x actions) ----
states = sorted(Q_dict.keys())
Q = np.stack([np.asarray(Q_dict[s], dtype=np.float32) for s in states], axis=0)

print("Loaded Q-table")
print("States:", Q.shape[0])
print("Actions:", Q.shape[1])
print("Q min / max:", Q.min(), Q.max())

# ============================================================
# 1) Full Q-table heatmap
# ============================================================
plt.figure(figsize=(12, 6))
plt.imshow(Q, aspect="auto")
plt.colorbar(label="Q-value")
plt.xlabel("Action index")
plt.ylabel("State (phase)")
plt.title("Q-table heatmap")
plt.tight_layout()
plt.show()

# ============================================================
# 2) Greedy policy (argmax action per state)
# ============================================================
greedy_actions = np.argmax(Q, axis=1)

plt.figure(figsize=(12, 2))
plt.imshow(greedy_actions[np.newaxis, :], aspect="auto")
plt.colorbar(label="Chosen action (argmax)")
plt.yticks([])
plt.xlabel("State (phase)")
plt.title("Greedy policy")
plt.tight_layout()
plt.show()

# ============================================================
# 3) Row-normalized Q-table (often clearer)
# ============================================================
Q_min = Q.min(axis=1, keepdims=True)
Q_ptp = Q.ptp(axis=1, keepdims=True) + 1e-8
Q_norm = (Q - Q_min) / Q_ptp

plt.figure(figsize=(12, 6))
plt.imshow(Q_norm, aspect="auto")
plt.colorbar(label="Normalized Q-value (per state)")
plt.xlabel("Action index")
plt.ylabel("State (phase)")
plt.title("Q-table heatmap (row-normalized)")
plt.tight_layout()
plt.show()
