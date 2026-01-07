from dolphin import event, gui, controller
from typing import TypedDict
import time
count=0

#actions
scale = 0.8

frames_per_action = 3

idle = controller.set_wiimote_acceleration(0,0,0,0)*scale

left = controller.set_wiimote_acceleration(0,-100,0,0)*scale
right = controller.set_wiimote_acceleration(0,100,0,0)*scale
up = controller.set_wiimote_acceleration(0,0,100,0)*scale
down = controller.set_wiimote_acceleration(0,0,-100,0)*scale
forward = controller.set_wiimote_acceleration(0,0,0,100)*scale
backward = controller.set_wiimote_acceleration(0,0,0,-100)*scale

up_left = controller.set_wiimote_acceleration(0,-100,100,0)*scale
up_right = controller.set_wiimote_acceleration(0,100,100,0)*scale
down_left = controller.set_wiimote_acceleration(0,-100,-100,0)*scale
down_right = controller.set_wiimote_acceleration(0,100,-100,0)*scale
forward_up = controller.set_wiimote_acceleration(0,0,100,100)*scale
forward_down = controller.set_wiimote_acceleration(0,0,-100,100)*scale
forward_left = controller.set_wiimote_acceleration(0,-100,0,100)*scale
forward_right = controller.set_wiimote_acceleration(0,100,0,100)*scale
backward_up = controller.set_wiimote_acceleration(0,0,100,-100)*scale
backward_down = controller.set_wiimote_acceleration(0,0,-100,-100)*scale
backward_left = controller.set_wiimote_acceleration(0,-100,0,-100)*scale
backward_right = controller.set_wiimote_acceleration(0,100,0,-100)*scale

forward_up_left = controller.set_wiimote_acceleration(0,-100,100,100)*scale
forward_up_right = controller.set_wiimote_acceleration(0,100,100,100)*scale
forward_down_left = controller.set_wiimote_acceleration(0,-100,-100,100)*scale
forward_down_right = controller.set_wiimote_acceleration(0,100,-100,100)*scale
backward_up_left = controller.set_wiimote_acceleration(0,-100,100,-100)*scale
backward_up_right = controller.set_wiimote_acceleration(0,100,100,-100)*scale
backward_down_left = controller.set_wiimote_acceleration(0,-100,-100,-100)*scale
backward_down_right = controller.set_wiimote_acceleration(0,100,-100,-100)*scale

#controls
a_press = controller.set_wiimote_buttons(0, {"A": True})
a_release = controller.set_wiimote_buttons(0, {"A": False})

while True:
    await event.frameadvance()
    count+=1


    if count%frames_per_action==0:

        print(controller.get_wiimote_acceleration(0))
    #controller.set_wiimote_acceleration(0, {})
        controller.set_wiimote_buttons(0, {"A": True})
    else:
        controller.set_wiimote_buttons(0, {"A": False})