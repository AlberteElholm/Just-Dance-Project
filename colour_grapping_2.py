import mss
import time
from pynput import mouse
m = mouse.Controller()

#Pixel value for reading reward
Pixel_x = 287  
Pixel_y = 422 

cd_frames = 6

armed = True

moves = 0
frame = 0
total_frames = 0
prev_label = "None"
goodcounter = 0
perfectcounter = 0

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

def reset_reward_state():
    global moves, total_frames
    moves = 0
    total_frames = 0

def reward_detector():
    global moves, frame, total_frames, prev_label, goodcounter, perfectcounter, Pixel_x, Pixel_y, cd_frames, armed

    frame += 1
    total_frames += 1
    r, g, b = read_rgb(Pixel_x, Pixel_y)
    label = classify_pixel(r, g, b)
    cooldown_ok = frame >= cd_frames
    is_judgement = label[0] in {"X", "OK", "Good", "Perfect", "Yeah"}
    label_changed = prev_label != label[0]
    is_clear = (label[0] == "None") or label_changed
    prev_label = label[0]
    if is_clear:
        armed = True
    # Fire event on first appearance
    if armed and is_judgement and cooldown_ok:
        moves += 1
        armed = False
        reward = label[1]
        if label[0] == "Good":
            goodcounter +=1
        if label[0] == "Perfect":
            perfectcounter +=1
        print(label[0],"moves:", moves, "total_frames:",total_frames, "good:",goodcounter,"Perfect:",perfectcounter)
        frame = 0
        return float(reward),moves,total_frames
    else:
        return None,moves,total_frames
    

# Helps to find the right pixel to read:
#while True:
#    print(m.position,read_rgb(m.position[0],m.position[1]),classify_pixel(read_rgb(m.position[0],m.position[1])[0],read_rgb(m.position[0],m.position[1])[1],read_rgb(m.position[0],m.position[1])[2]))
#    time.sleep(0.3)