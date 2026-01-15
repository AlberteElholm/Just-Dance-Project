# q_table_heatmap_plus_actions.py
# Copy–paste and run

import pickle
import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
PKL_PATH = "agent.pkl"   # change if needed
# Optional: give your 27 actions names (otherwise indices are used)
ACTION_NAMES = None
# Example:
# ACTION_NAMES = [f"A{i}" for i in range(27)]
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

n_states, n_actions = Q.shape

if ACTION_NAMES is None:
    ACTION_NAMES = [str(i) for i in range(n_actions)]
else:
    if len(ACTION_NAMES) != n_actions:
        raise ValueError(f"ACTION_NAMES has length {len(ACTION_NAMES)} but Q has {n_actions} actions")

print("Loaded Q-table")
print("States:", n_states)
print("Actions:", n_actions)
print("Q min / max:", float(Q.min()), float(Q.max()))

# ============================================================
# 1) Full Q-table heatmap
# ============================================================
plt.figure(figsize=(12, 6))
plt.imshow(Q, aspect="auto", interpolation="none")
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
plt.imshow(greedy_actions[np.newaxis, :], aspect="auto", interpolation="none")
plt.colorbar(label="Chosen action (argmax)")
plt.yticks([])
plt.xlabel("State (phase)")
plt.title("Greedy policy (best action per state)")
plt.tight_layout()
plt.show()

# ============================================================
# 3) Row-normalized Q-table (often clearer)
# ============================================================
Q_min = Q.min(axis=1, keepdims=True)
Q_ptp = np.ptp(Q, axis=1, keepdims=True) + 1e-8
Q_norm = (Q - Q_min) / Q_ptp

plt.figure(figsize=(12, 6))
plt.imshow(Q_norm, aspect="auto", interpolation="none")
plt.colorbar(label="Normalized Q-value (per state)")
plt.xlabel("Action index")
plt.ylabel("State (phase)")
plt.title("Q-table heatmap (row-normalized)")
plt.tight_layout()
plt.show()

# ============================================================
# 4) Sum of Q-values per action (ascending)
# ============================================================
Q_action_sum = Q.sum(axis=0)

order_sum = np.argsort(Q_action_sum)  # ascending

print("\nSum of Q-values per action (ascending):")
for a in order_sum:
    print(f"Action {a:2d} ({ACTION_NAMES[a]}): {Q_action_sum[a]: .3f}")

plt.figure(figsize=(12, 3))
plt.bar(range(n_actions), Q_action_sum[order_sum])
plt.xticks(range(n_actions),
           [ACTION_NAMES[a] for a in order_sum],
           rotation=90)
plt.xlabel("Action (sorted)")
plt.ylabel("Sum of Q-values")
plt.title("Action preference (sum over states, ascending)")
plt.tight_layout()
plt.show()


# ============================================================
# 5) Mean Q-value per action (ascending)
# ============================================================
Q_action_mean = Q.mean(axis=0)

order_mean = np.argsort(Q_action_mean)  # ascending

print("\nMean Q-value per action (ascending):")
for a in order_mean:
    print(f"Action {a:2d} ({ACTION_NAMES[a]}): {Q_action_mean[a]: .3f}")

plt.figure(figsize=(12, 3))
plt.bar(range(n_actions), Q_action_mean[order_mean])
plt.xticks(range(n_actions),
           [ACTION_NAMES[a] for a in order_mean],
           rotation=90)
plt.xlabel("Action (sorted)")
plt.ylabel("Mean Q-value")
plt.title("Action preference (mean over states, ascending)")
plt.tight_layout()
plt.show()

# ============================================================
# 6) Greedy action frequency (ascending)
# ============================================================
unique, counts = np.unique(greedy_actions, return_counts=True)
freq = np.zeros(n_actions, dtype=int)
freq[unique] = counts

order_freq = np.argsort(freq)  # ascending

print("\nGreedy action frequency (ascending):")
for a in order_freq:
    print(f"Action {a:2d} ({ACTION_NAMES[a]}): {freq[a]}")

plt.figure(figsize=(12, 3))
plt.bar(range(n_actions), freq[order_freq])
plt.xticks(range(n_actions),
           [ACTION_NAMES[a] for a in order_freq],
           rotation=90)
plt.xlabel("Action (sorted)")
plt.ylabel("Number of states")
plt.title("Greedy policy action frequency (ascending)")
plt.tight_layout()
plt.show()