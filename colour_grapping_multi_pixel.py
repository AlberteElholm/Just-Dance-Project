import time
import mss
import numpy as np
import colorsys
from pynput import mouse

m = mouse.Controller()

PIXEL_X = 280  # girlfrend
PIXEL_Y = 446

POLL_HZ = 248                 # how often we sample the pixel
ARM_ON_UNKNOWN = False        # keep False to avoid noisy triggers
DEBUG = False                 # set False when done

# Beat-based re-arm settings
BPM = 124
SPEED_MULTIPLIER = 10         # speed up (>1.0) or slow down (<1.0) the rearm timing
COOLDOWN_MS = 500 / SPEED_MULTIPLIER

# -----------------------
# Pixel reader (robust): median RGB over a small region
# -----------------------
sct = mss.mss()

def read_rgb_region(x, y, size=3):
    """
    Read a size x size region centered at (x,y) and return median RGB.
    This is much more stable than a single pixel due to anti-aliasing/UI jitter.
    """
    half = size // 2
    img = sct.grab({"top": y - half, "left": x - half, "width": size, "height": size})
    arr = np.array(img)[:, :, :3]           # BGRA -> BGR (first 3)
    rgb = arr[:, :, ::-1].reshape(-1, 3)    # BGR -> RGB, flatten
    med = np.median(rgb, axis=0)
    return int(med[0]), int(med[1]), int(med[2])

def read_rgb(x, y):
    # Backwards-compatible wrapper; switch size to 5 if you want even more smoothing.
    return read_rgb_region(x, y, size=3)

# -----------------------
# Pixel classifier (improved): HSV with saturation/value gates
# -----------------------
def classify_pixel(r, g, b):
    """
    Returns: ["Label", reward]
    Labels: None, X, Yeah, Perfect, Good, OK, Unknown

    Uses HSV hue ranges + saturation/value gates for robustness under brightness shifts.
    """
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rn, gn, bn)
    H = h * 360.0

    # 1) Background / no message
    # - Very dark OR dark+low-sat: treat as None (message gone)
    if v < 0.18 or (v < 0.25 and s < 0.25):
        return ["None", 0.0]

    # 2) If it's too desaturated, avoid false positives (white/gray UI)
    if s < 0.35:
        return ["Unknown", 0.0]

    # 3) Hue-based classification
    # Red (X): around 0°, wraps around 360°
    if (H <= 20 or H >= 340) and v > 0.25:
        return ["X", 0.0]

    # Yellow/Gold (Yeah): around 38°–70°
    if 38 <= H <= 70 and v > 0.35:
        return ["Yeah", 0.0]

    # Green (Perfect): around 85°–160°
    if 85 <= H <= 160 and v > 0.30:
        return ["Perfect", 1.0]

    # Blue (Good): around 190°–250°
    if 190 <= H <= 250 and v > 0.30:
        return ["Good", 0.5]

    # Purple (OK): around 255°–320°
    if 255 <= H <= 320 and v > 0.30:
        return ["OK", 0.3]

    return ["Unknown", 0.0]

# -----------------------
# EVENT-DRIVEN DETECTOR
# -----------------------
def detect_point_events():
    """
    Generator that yields an event each time a new point message appears.

    Event definition (rising edge):
      None/Unknown -> (X/OK/Good/Perfect/Yeah)

    Re-arming:
      After firing, we won't fire again until we see label == "None"
      (i.e., message disappeared).
    """
    dt = 1.0 / POLL_HZ  # time step between samples

    armed = True
    last_event_time = 0.0
    last_t = None

    # Optional: require stability for N consecutive samples to reduce flicker triggers
    STABLE_N = 2
    stable_count = 0
    last_label_name = None

    while True:
        r, g, b = read_rgb(PIXEL_X, PIXEL_Y)
        label = classify_pixel(r, g, b)

        now = time.time()
        cooldown_ok = (now - last_event_time) * 1000.0 >= COOLDOWN_MS

        is_judgement = label[0] in {"X", "OK", "Good", "Perfect", "Yeah"}
        is_clear = (label[0] == "None") or (ARM_ON_UNKNOWN and label[0] == "Unknown")

        # Re-arm when message is gone
        if is_clear:
            armed = True
            stable_count = 0
            last_label_name = None

        # Stability filter (helps with one-frame color flickers)
        if label[0] == last_label_name:
            stable_count += 1
        else:
            last_label_name = label[0]
            stable_count = 1

        stable_ok = stable_count >= STABLE_N

        # Fire event on first stable appearance
        if armed and is_judgement and cooldown_ok and stable_ok:
            event_dt = 0.0 if last_t is None else (now - last_t)
            last_t = now
            last_event_time = now
            armed = False

            yield {
                "t": now,
                "dt": event_dt,        # time since previous event
                "label": label,
                "rgb": (r, g, b),
            }

        if DEBUG:
            print(f"RGB={r,g,b} label={label} armed={armed} stable={stable_count}")

        time.sleep(dt)

# -----------------------
# DEMO / TEST RUN
# -----------------------
count = 0

if __name__ == "__main__":
    print("Listening for new point-message events... (Ctrl+C to stop)")
    for ev in detect_point_events():
        print(
            f"EVENT: label={ev['label']} dt={ev['dt']:.3f}s "
            f"rgb={ev['rgb']} count={count}"
        )
        count += 1
