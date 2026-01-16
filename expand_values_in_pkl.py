import pickle
import numpy as np
from pathlib import Path

IN_PATH  = Path("agent_100_1d.pkl")          # your existing file
OUT_PATH = Path("agent_100_1d_2.pkl") # new file (safer)

OLD_N = 7
NEW_N = 18  # = 7 + 11

with open(IN_PATH, "rb") as f:
    data = pickle.load(f)

if not isinstance(data, dict) or "Q" not in data:
    raise TypeError("Expected a dict with a 'Q' key in the pickle file")

Q = data["Q"]
if not isinstance(Q, dict):
    raise TypeError(f"Expected data['Q'] to be a dict, got {type(Q)}")

changed = 0
skipped = 0

for k, v in Q.items():
    arr = np.asarray(v, dtype=np.float32)

    if arr.ndim != 1:
        raise ValueError(f"State {k}: expected 1D vector, got shape {arr.shape}")

    if arr.size == NEW_N:
        skipped += 1
        continue

    if arr.size != OLD_N:
        raise ValueError(f"State {k}: expected length {OLD_N} (or {NEW_N}), got {arr.size}")

    Q[k] = np.concatenate([arr, np.zeros(NEW_N - OLD_N, dtype=np.float32)])
    changed += 1

data["Q"] = Q

with open(OUT_PATH, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Done. Expanded {changed} states. Skipped {skipped} already-expanded states.")
print(f"Wrote: {OUT_PATH}")