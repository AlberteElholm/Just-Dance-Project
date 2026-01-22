from pynput import mouse
from colour_grab import read_rgb, classify_pixel
import time

m = mouse.Controller()
# Helps to find the right pixel to read:
# prints: pixel position (x,y) (r,g,b) [label,reward]
while True:
    print(m.position,read_rgb(m.position[0],m.position[1]),classify_pixel(read_rgb(m.position[0],m.position[1])[0],read_rgb(m.position[0],m.position[1])[1],read_rgb(m.position[0],m.position[1])[2]))
    time.sleep(0.3)