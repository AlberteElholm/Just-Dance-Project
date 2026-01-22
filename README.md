# Just Dance Project

# Downloads:
1. Download this repository.
2. Download Just dance 2 for the Wii.
3. Download Felk's version of Dolphin emulator https://github.com/Felk/dolphin. 
Under releases -> Assets. THere should be a file named dolphin-scripting-preview4-x64.7z

# Initial setup:
1. Open config from this repository in a code editor such as VSCode.
2. Find the paths for Just dance 2 and Felk's version of Dolphin, and replace the existing paths with yours in config.
3. Open Felk's version of Dolphin. (dolphin-scripting-preview4-x64\Dolphin.exe)
4. In Dolphin, find the path for Just dance 2 and add it by clicking on the right side.
5. Right-click on Just dance 2, and choose properties -> Game Config -> Editor -> Under Default config (SD2.ini), press Presets -> Edtior -> Open in External Editor. Write this under [Core]: (EmulationSpeed = 10.0). You can change this value.
6. Open controllers -> Configure Wii remote 1 -> General and Options -> Extention -> From Nunchuck to None.
7. Close Dolphin

# Running the Game:
You can control the wii game with your curser as pointer, left-click is A-button and right-click is the B-button.
1. Run Master.py
2. Make the game full screen
3. In the song selection menu, change the song length to short in the top left corner.
4. Find "Girlfriend" By Avril Lavigne
5. Choose the song, click past "Press 'A' to skip" and hold right-click to start the song and learning. (This step sometimes fails. You can either just close the game and try again, or wait a second or two, and left-click to start it anyway. It will correct itself after a song or two.)
It should now be running and be able to reset automatically.

# When running:
keep your mouse on the game, and don't click, as it can disturb the learning.

If you have two screens, you can look at your results update live.

# Possible Issues:
Finding the right pixel. We have used a 1920x1080 screen on windows 11. If you have a different resolution or OS, you need to manually find a pixel that works. We've made a tool to help you find the right pixel, called pixel_finder.py. You need to find a pixel position (x,y), that includes all colour values when you recieve a reward.

If you are training an existing agent, you need to know what values the config had when it was first changed, and match them.

If you want to train on a new song, you need to change song_moves = 97, total_song_frames = 5100, aswell as finding the correct pixel, as it usually changes vertically between songs.