import time
import mss
from pynput import mouse
m = mouse.Controller()

PIXEL_X = 280 #280 rasputin  
PIXEL_Y = 446 #553  

POLL_HZ = 248                 # how often we sample the pixel
ARM_ON_UNKNOWN = False       # keep False to avoid noisy triggers
DEBUG = False                 # set False when done

# Beat-based re-arm settings
BPM = 124                    # song tempo
SPEED_MULTIPLIER = 10       # speed up (>1.0) or slow down (<1.0) the rearm timing
COOLDOWN_MS = 500/SPEED_MULTIPLIER            # We primarily rely on re-arming (message disappears) rather than cooldown.

#Pixel reader
sct = mss.mss()
def read_rgb(x, y):
    img = sct.grab({"top": y, "left": x, "width": 1, "height": 1})
    r, g, b = img.pixel(0, 0)
    return r, g, b

#Pixel classifier
def classify_pixel(r, g, b):
    if r < 45 and g < 50 and b < 50:
        return ["None",0]
    if r > 120 and g < 60 and b < 60:
        return ["X",0]          # red
    if r > 200 and g > 150 and b < 80:
        return ["Yeah",0]       # gold/yellow
    if g > 170 and r > 100:
        return ["Perfect",1.0]   # green
    if b > 200 and r < 20:
        return ["Good",0.5]       # blue
    if r > 120 and b > 120:
        return ["OK",0.3]         # purple
    return ["Unknown",0]    

#for i in range (1000):
#    print(m.position,read_rgb(m.position[0],m.position[1]),classify_pixel(read_rgb(m.position[0],m.position[1])[0],read_rgb(m.position[0],m.position[1])[1],read_rgb(m.position[0],m.position[1])[2]))
#    time.sleep(0.3)
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
    dt = 1.0 / POLL_HZ #time step between samples

    armed = True
    last_event_time = 0.0
    last_t = None

    while True:
        r, g, b = read_rgb(PIXEL_X, PIXEL_Y)
        label = classify_pixel(r, g, b)

        now = time.time()
        cooldown_ok = (now - last_event_time) * 1000.0 >= COOLDOWN_MS

        is_judgement = label[0] in {"X", "OK", "Good", "Perfect", "Yeah"}
        is_clear = (label[0] == "None") or (ARM_ON_UNKNOWN and label[0] == "Unknown")

        # Re-arm when message is gone or after rearm_frames time has passed
        rearm_threshold_ms = 1/((124/60)*SPEED_MULTIPLIER)
        if is_clear: #or ((now - last_event_time)>= rearm_threshold_ms):
            armed = True

        # Fire event on first appearance
        if armed and is_judgement and cooldown_ok:
            event_dt = 0.0 if last_t is None else (now - last_t)
            last_t = now
            last_event_time = now
            armed = False

            yield {
                "t": now,
                "dt": event_dt,         # time since previous event
                "label": label,
                "rgb": (r, g, b),
            }

        if DEBUG:
            # Useful during calibration. Turn off later.
            print(f"RGB={r,g,b}  label={label}  armed={armed}")

        time.sleep(dt)


# -----------------------
# DEMO / TEST RUN
# -----------------------
count=0

if __name__ == "__main__":
   print("Listening for new point-message events... (Ctrl+C to stop)")
   for ev in detect_point_events():
        label, r = ev["label"]
        print(f"EVENT: label={ev['label']} dt={ev['dt']:.3f}s rgb={ev['rgb']} count: {count}")
        count+=1
