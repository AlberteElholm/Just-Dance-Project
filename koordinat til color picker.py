from pynput import mouse
import time

time.sleep(5)

m = mouse.Controller()
print(m.position)
