from dolphin import event, controller
import time
import random

SCALE = 0.8
USE_NEARBY = True

REWARD = {"Perfect": 1.0, "Good": 0.5, "OK": 0.3, "X": 0.0, "Yeah": 0.0}

ACTIONS = {
    0: ((0,   0,   0),   (0, 0, 0)),   # idle

    1: ((-100, 0,   0),  (-1, 0, 0)),  # left
    2: ((100,  0,   0),  ( 1, 0, 0)),  # right
    3: ((0,   100,  0),  (0,  1, 0)),  # up
    4: ((0,  -100,  0),  (0, -1, 0)),  # down
    5: ((0,    0, 100),  (0, 0,  1)),  # forward
    6: ((0,    0,-100),  (0, 0, -1)),  # backward

    7:  ((-100, 100,  0), (-1, 1, 0)),  # up_left
    8:  ((100,  100,  0), ( 1, 1, 0)),  # up_right
    9:  ((-100,-100,  0), (-1,-1, 0)),  # down_left
    10: ((100, -100,  0), ( 1,-1, 0)),  # down_right

    11: ((0,   100, 100), (0, 1,  1)),  # forward_up
    12: ((0,  -100, 100), (0,-1,  1)),  # forward_down
    13: ((-100, 0,  100), (-1,0,  1)),  # forward_left
    14: ((100,  0,  100), ( 1,0,  1)),  # forward_right

    15: ((0,   100,-100), (0, 1, -1)),  # backward_up
    16: ((0,  -100,-100), (0,-1, -1)),  # backward_down
    17: ((-100, 0,-100),  (-1,0, -1)),  # backward_left
    18: ((100,  0,-100),  ( 1,0, -1)),  # backward_right
}
N_ACTIONS = len(ACTIONS)

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def chebyshev_dist(a, b):
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]), abs(a[2]-b[2]))

def allowed_next_actions(prev_action: int, use_nearby: bool = True):
    prev_vec = ACTIONS[prev_action][1]
    allowed = []
    for a, (_, v) in ACTIONS.items():
        if a == 0:
            allowed.append(a)
            continue
        if use_nearby and chebyshev_dist(v, prev_vec) > 1:
            continue
        if dot(prev_vec, v) < 0:
            continue
        allowed.append(a)
    return allowed

def apply_action(action_id: int):
    (ax, ay, az), _ = ACTIONS[action_id]
    controller.set_wiimote_acceleration(
        0,
        int(ax * SCALE),
        int(ay * SCALE),
        int(az * SCALE),
    )

def choose_action():
    # placeholder; replace with your agent output
    return random.randrange(N_ACTIONS)

# Plug in your score detector object here:
# score_detector.poll_event() -> None or {"label": "...", "t": ...}
score_detector = None  # TODO: set this

prev_action = 0
segment_actions = []
segment_start_t = time.time()

while True:
    await event.frameadvance()  # one action per frame (~60/sec)

    proposed = choose_action()

    allowed = allowed_next_actions(prev_action, use_nearby=USE_NEARBY)
    if proposed not in allowed:
        proposed = random.choice(allowed)

    apply_action(proposed)
    prev_action = proposed
    segment_actions.append((time.time(), proposed))
    print(controller.get_wiimote_acceleration(0))
    ev = score_detector.poll_event() if score_detector is not None else None
    if ev is not None:
        label = ev["label"]
        reward = REWARD.get(label, 0.0)

        segment_end_t = time.time()
        print(
            f"EVENT {label} reward={reward} "
            f"actions={len(segment_actions)} "
            f"duration={segment_end_t - segment_start_t:.3f}s"
        )
        

        # reset segment
        segment_actions = []
        segment_start_t = time.time()