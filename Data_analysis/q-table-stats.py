import pickle
import numpy as np
from pathlib import Path
import sys

FOLDER = Path("q-tables")  # <-- change this

def describe_pkl(path: Path):
    size_mb = path.stat().st_size / (1024 ** 2)

    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        return f"{path.name}: FAILED TO LOAD ({e})"

    lines = []
    lines.append(f"{path.name}")
    lines.append(f"  file size: {size_mb:.2f} MB")
    lines.append(f"  type: {type(obj).__name__}")

    # --- Episode count ---
    episode_count = None
    if isinstance(obj, dict):
        episode_count = obj.get("episode_count", None)

    if episode_count is not None:
        lines.append(f"  episodes run: {episode_count}")
    else:
        lines.append("  episodes run: (not stored)")

    # --- Q-table inspection ---
    if isinstance(obj, dict) and "Q" in obj:
        Q = obj["Q"]
        if isinstance(Q, dict) and len(Q) > 0:
            first_val = next(iter(Q.values()))
            if isinstance(first_val, np.ndarray):
                n_states = len(Q)
                action_shape = first_val.shape
                total_q = n_states * np.prod(action_shape)
                lines.append("  Q-table:")
                lines.append(f"    states: {n_states}")
                lines.append(f"    action shape: {action_shape}")
                lines.append(f"    total Q-values: {total_q}")
            else:
                lines.append("  Q-table values are not numpy arrays")
        else:
            lines.append("  Q-table empty or invalid")

    elif isinstance(obj, np.ndarray):
        lines.append(f"  array shape: {obj.shape}")
        lines.append(f"  elements: {obj.size}")

    else:
        lines.append("  (no structured inspection available)")

    return "\n".join(lines)

def main():
    if not FOLDER.exists():
        print("Folder does not exist:", FOLDER)
        sys.exit(1)

    pkls = sorted(FOLDER.glob("*.pkl"))
    if not pkls:
        print("No .pkl files found in", FOLDER)
        return

    print(f"Found {len(pkls)} .pkl files\n")
    for pkl in pkls:
        print(describe_pkl(pkl))
        print("-" * 50)

if __name__ == "__main__":
    main()
