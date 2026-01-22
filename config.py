# Paths
Felk_dolphin_exe = r"C:/Users/esben/Downloads/dolphin-scripting-preview4-x64/dolphin"
Game_path    = r"C:/Users/esben/Downloads/dolphin-2512-x64/Dolphin-x64/spil/Just_dance2.wbfs"

# RL parameters
greedy = False #if true, changes policy from epsilon-greedy to greedy.
n_actions = 7 #1d = 7 actions, 2d = 19 actions, 3d = 27 actions

alpha = 0.05
gamma = 0.95
lam = 0.90

eps_start = 1.0
eps_min = 0.05
eps_decay = 0.995

# Environment / game settings
base = 100 # acceleration 100=10m/s^2
frames_per_action = 5 # amount of frames before a new action is taken
song_moves = 97 #233 for the long version
total_song_frames = 5100
total_song_phases = round(total_song_frames/frames_per_action)
# Files, if no file it will create one
agent_path = "q-tables/100_1d.pkl"
log_path = "results/4_frames_new.pkl"

#Pixel value for reading reward, works for 1920x1080 screens. If this doesn't work, use the inial_pixel_finder.py tool.
Pixel_x = 287  
Pixel_y = 422

#slave
base = 100.0  # desired acceleration magnitude
