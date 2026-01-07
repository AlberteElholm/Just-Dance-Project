import pyautogui
import time
from dolphin import event

async def main():
    while True:
        await event.frameadvance()
        print('script is running')
# ---- FAIL-SAFE ----
# Moving the mouse to the top-left corner aborts the script
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01  # small delay between actions

# ---- BUTTON PRESS ----
def tap(key, duration=0.05):
    """Press and release a key quickly."""
    pyautogui.keyDown(key)
    time.sleep(duration)
    pyautogui.keyUp(key)

# ---- BUTTON HOLD ----
def hold(key, hold_time):
    """Hold a key for a given amount of time (seconds)."""
    pyautogui.keyDown(key)
    time.sleep(hold_time)
    pyautogui.keyUp(key)

# ---- EXAMPLE USAGE ----
try:
    while True:
        tap('a')          # quick press of 'a'
        time.sleep(1)
        hold('w', 2.0)    # hold 'w' for 2 seconds
except pyautogui.FailSafeException:
    print("Fail-safe triggered. Script stopped.")
