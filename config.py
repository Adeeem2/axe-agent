import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

HIGHWAY_REGION = {
    "top":    300,
    "left":   430,
    "width":  500,
    "height": 400,
}

STRIKEBAR_Y = 300

SUSTAIN_LOOKAHEAD_Y = 200

HEALTH_REGION = {
    "top":    526,   # Y position (second number)
    "left":   1041,  # X position (first number)
    "width":  112,
    "height": 20
}

STATS_REGION = {
    "top":    193,
    "left":   828,
    "width":  195,
    "height": 147,
}

LANE_COLORS_BGR = {
    0: (3,   128,  3),
    1: (0,   0,    164),
    2: (57,  183,  183),
    3: (153, 82,   0),
    4: (0,   135,  192),
}

COLOR_TOLERANCE = 60

LANE_KEYS = {
    0: 'q',
    1: 's',
    2: 'j',
    3: 'k',
    4: 'l',
}

STRUM_KEY  = 'down'
REPEAT_KEY = 'h'
REPEAT_INTERVAL = 300

STATS_READ_INTERVAL = 200

# ─────────────────────────────────────────────────────────────
# NEW: Training improvements
# ─────────────────────────────────────────────────────────────

# Frame stacking - use SB3's VecFrameStack instead of manual deque
USE_VEC_FRAME_STACK = True
N_FRAME_STACK = 4

# Reward shaping
REWARD_SCORE_SCALE = 0.002      # per score point
REWARD_GHOST_PENALTY = 0.3      # per ghost note
REWARD_FAIL_PENALTY = -10.0     # song failure
REWARD_SURVIVAL_BONUS = 0.001   # per step alive (dense reward)
REWARD_HIT_BONUS = 0.05         # bonus when hit% improves
REWARD_COMBO_BONUS = 0.01       # per combo milestone

# Curriculum learning
CURRICULUM_ENABLED = True
CURRICULUM_STAGES = [
    {"name": "easy",     "min_nps": 0,   "max_nps": 3.0, "min_steps": 50000},
    {"name": "medium",   "min_nps": 3.0, "max_nps": 6.0, "min_steps": 100000},
    {"name": "hard",     "min_nps": 6.0, "max_nps": 10.0, "min_steps": 200000},
    {"name": "expert",   "min_nps": 10.0, "max_nps": 999, "min_steps": 500000},
]

# Checkpoint saving - synchronous to avoid pydirectinput thread issues
SAVE_EVERY_N_STEPS = 4096
SAVE_SYNCHRONOUS = True  # Set True to save in main thread (safer for pydirectinput)

# Admin check
REQUIRE_ADMIN = True

# Observation preprocessing
OBS_GRAYSCALE = False  # Keep RGB for color info (lane colors)
OBS_RESIZE = (84, 84)

# Action smoothing - prevent rapid toggle
ACTION_MIN_HOLD_FRAMES = 2  # Minimum frames to hold a key before release
