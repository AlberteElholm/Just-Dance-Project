# SimpleDolphinEnv.py
# Master process: runs outside Dolphin
# - Listens on localhost:26330
# - Receives (obs_bytes, reward, done, trunc, info) from DolphinScript
# - Sends ("ACT", action) and ("RESET", 0)

from multiprocessing.connection import Listener
import numpy as np
import time


HOST = "localhost"
PORT = 26330
AUTHKEY = b"secret password"

OBS_W = 140
OBS_H = 75
OBS_SHAPE = (OBS_H, OBS_W)  # (75, 140)


class SimpleDolphinEnv:
    def __init__(self):
        self.listener = Listener((HOST, PORT), authkey=AUTHKEY)
        print(f"[master] Listening on {(HOST, PORT)} ...")
        self.conn = self.listener.accept()
        print("[master] Dolphin connected.")

        # handshake
        msg = self.conn.recv()
        if not (isinstance(msg, tuple) and len(msg) >= 1 and msg[0] == "READY"):
            raise RuntimeError(f"[master] Handshake failed. Got: {msg}")
        print("[master] Handshake OK.")

    def reset(self):
        self.conn.send(("RESET", 0))
        msg = self.conn.recv()
        obs = self._decode_obs_msg(msg)
        return obs

    def step(self, action: int):
        self.conn.send(("ACT", int(action)))
        msg = self.conn.recv()
        obs, reward, done, trunc, info = self._decode_step_msg(msg)
        return obs, reward, done, trunc, info

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
        try:
            self.listener.close()
        except Exception:
            pass

    def _decode_obs_msg(self, msg):
        # Expected: ("OBS", obs_bytes)
        if not (isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "OBS"):
            raise RuntimeError(f"[master] Bad OBS msg: {msg}")

        obs_bytes = msg[1]
        obs = np.frombuffer(obs_bytes, dtype=np.uint8).reshape(OBS_SHAPE)
        return obs

    def _decode_step_msg(self, msg):
        # Expected: ("STEP", obs_bytes, reward, done, trunc, info_dict)
        if not (isinstance(msg, tuple) and len(msg) == 6 and msg[0] == "STEP"):
            raise RuntimeError(f"[master] Bad STEP msg: {msg}")

        obs_bytes = msg[1]
        reward = float(msg[2])
        done = bool(msg[3])
        trunc = bool(msg[4])
        info = msg[5] if isinstance(msg[5], dict) else {}

        obs = np.frombuffer(obs_bytes, dtype=np.uint8).reshape(OBS_SHAPE)
        return obs, reward, done, trunc, info


if __name__ == "__main__":
    env = SimpleDolphinEnv()

    try:
        obs = env.reset()
        print("[master] Reset OK. First obs shape:", obs.shape)

        # Simple connectivity test: send a few actions
        test_actions = [0, 1, 2, 0, 19, 0]  # idle, left, right, idle, forward_up_left, idle
        for a in test_actions:
            obs, r, d, t, info = env.step(a)
            print(f"[master] a={a:2d}  r={r:.2f}  done={d}  trunc={t}  last_label={info.get('last_label')}")

            # slow down so you can see behavior
            time.sleep(0.1)

    finally:
        env.close()
        print("[master] Closed.")
