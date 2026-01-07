from dolphin import event, gui
import sys
red = 0xffff0000
frame_counter = 0

print(f"{sys.executable}")
print(f"{sys.version}")
while True:
    await event.frameadvance()
    frame_counter += 1
    # draw on screen
    gui.draw_text((10, 10), red, f"Frame: {sys.version}")
    gui.draw_text((10, 20), red, f"Frame: {sys.executable}")
    # print to console
    if frame_counter % 60 == 0:
        pass#print(f"The frame count has reached {frame_counter}")