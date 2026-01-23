import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # saves PNG (no pop-up window)
import matplotlib.pyplot as plt
from pathlib import Path

# ===== CHANGE THIS =====
PKL_PATH = Path(r"C:\Users\esben\OneDrive\Documents\GitHub\Just-Dance-Project\q-tables\100_1d.pkl")
# ======================

# Force states to be exactly 0..1250 (inclusive)
TARGET_STATES = 1251  # 0..1250

OUT_PATH = PKL_PATH.with_name(PKL_PATH.stem + "_actions_x_states_0-1250.png")

def load_q(obj):
    """
    Supports:
      1) {"Q": {...}, "episode_count": ...}
      2) Q-table dict directly {state:int -> np.array}
    """
    if isinstance(obj, dict) and "Q" in obj and isinstance(obj["Q"], dict):
        return obj["Q"], obj.get("episode_count", None)

    if isinstance(obj, dict) and len(obj) > 0:
        v0 = next(iter(obj.values()))
        if isinstance(v0, np.ndarray):
            return obj, None

    return None, None

def q_to_matrix_fixed_states(Q):
    """
    Build (states x actions) matrix for states 0..1250.
    - Crops states > 1250
    - Zero-pads missing states in range
    """
    # infer n_actions from any existing state in range (prefer 0 if present)
    if 0 in Q:
        sample = Q[0]
    else:
        # pick any value
        sample = next(iter(Q.values()))
    n_actions = int(np.asarray(sample).reshape(-1).shape[0])

    rows = []
    for s in range(TARGET_STATES):
        if s in Q:
            v = np.asarray(Q[s], dtype=np.float32).reshape(-1)
            if v.shape[0] != n_actions:
                raise ValueError(f"State {s} has {v.shape[0]} actions, expected {n_actions}")
            rows.append(v)
        else:
            rows.append(np.zeros(n_actions, dtype=np.float32))

    return np.vstack(rows)  # (1251, n_actions)

def main():
    print("PKL_PATH:", PKL_PATH)
    print("Exists?:", PKL_PATH.exists())
    print("Will save to:", OUT_PATH)

    if not PKL_PATH.exists():
        raise FileNotFoundError(f"Pickle not found: {PKL_PATH}")

    with open(PKL_PATH, "rb") as f:
        obj = pickle.load(f)

    Q, ep = load_q(obj)
    if Q is None:
        if isinstance(obj, dict):
            raise RuntimeError("No Q-table found. Top-level keys: " + str(list(obj.keys())[:30]))
        raise RuntimeError(f"No Q-table found. Top-level type: {type(obj)}")

    mat_sa = q_to_matrix_fixed_states(Q)  # (states, actions) = (1251, n_actions)
    mat_as = mat_sa.T                     # (actions, states)

    print("Matrix shapes:")
    print("  (states, actions):", mat_sa.shape)
    print("  (actions, states):", mat_as.shape)
    if ep is not None:
        print("episode_count:", ep)

    # Figure sizing: width scales with number of states (1251 columns)
    plt.figure(figsize=(max(8, mat_as.shape[1] / 90), max(4, mat_as.shape[0] / 2.5)))
    plt.imshow(mat_as, aspect="auto", interpolation="nearest")
    plt.xlabel("Phase index")
    plt.ylabel("Action index")
    title = f"Heatmap for q-table"
    if ep is not None:
        title += f" | ep={ep}"
    plt.title(title)
    plt.colorbar(label="Q value")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    plt.close()

    print("[OK] Saved:", OUT_PATH)
    print("Open the PNG in Finder to view the heatmap.")

if __name__ == "__main__":
    main()
