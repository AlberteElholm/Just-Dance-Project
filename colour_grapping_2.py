import mss
from pynput import mouse
m = mouse.Controller()

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
    if r > 120 and g < 60 and b < 50:
        return ["X",0]          # red 
    if r > 220 and g > 220 and b < 20:
        return ["Yeah",0]       # gold/yellow
    if g > 170 and r > 100 and b < 20:
        return ["Perfect",1.0]   # green
    if b > 200 and r < 20:
        return ["Good",0.5]       # blue
    if r > 200 and b > 200 and g<20:
        return ["OK",0.1]         # purple
    return ["Unknown",0]    

# Helps to find the right pixel to read:
#while True:
#    print(m.position,read_rgb(m.position[0],m.position[1]),classify_pixel(read_rgb(m.position[0],m.position[1])[0],read_rgb(m.position[0],m.position[1])[1],read_rgb(m.position[0],m.position[1])[2]))
#    time.sleep(0.3)