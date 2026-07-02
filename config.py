HIGHWAY_REGION = {
    "top":    300,
    "left":   430,
    "width":  500,
    "height": 400,
}

STRIKEBAR_Y = 340

SUSTAIN_LOOKAHEAD_Y = 200

# Thin strip right at the strumbar — relative to monitor origin, same as other regions.
# Derived from HIGHWAY_REGION + STRIKEBAR_Y so it always stays aligned.
HIT_ZONE_REGION = {
    "top":    HIGHWAY_REGION["top"]  + STRIKEBAR_Y - 15,  # centred on strikebar
    "left":   HIGHWAY_REGION["left"],
    "width":  HIGHWAY_REGION["width"],
    "height": 30,
}

# Region above the hitzone for detecting approaching notes (not yet at strum bar).
# Placed NOTE_ABOVE_OFFSET pixels above the hitzone so colors are read before
# they reach the hitzone, preventing false-positive note-present detections.
NOTE_ABOVE_OFFSET = 48
NOTE_DETECTION_REGION = {
    "top":    HIT_ZONE_REGION["top"] - NOTE_ABOVE_OFFSET,
    "left":   HIT_ZONE_REGION["left"] ,
    "width":  HIT_ZONE_REGION["width"] - 25,
    "height": 40,
}

# New reward values (replace the old ones)
REWARD_HIT          =  1.0   # correct fret + strum on note
REWARD_MISS_STRUM   = -0.3   # strummed with no note, or wrong timing
REWARD_WRONG_FRET   = -0.5   # note present but pressed wrong fret
REWARD_SUSTAIN_HOLD =  0.1   # holding correct fret during sustain tail
REWARD_SURVIVAL_BONUS = 0.0  # set to 0 — we don't want "do nothing" to be safe
REWARD_FAIL_PENALTY = -2.0


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
SAVE_EVERY_N_STEPS = 20000
SAVE_SYNCHRONOUS = True  # Set True to save in main thread (safer for pydirectinput)

# Admin check
REQUIRE_ADMIN = True

# Observation preprocessing
OBS_GRAYSCALE = False  # Keep RGB for color info (lane colors)
OBS_RESIZE = (84, 84)

# ─────────────────────────────────────────────────────────────
# NEW: CloneHero Specific Tuning (from monolithic env)
# ─────────────────────────────────────────────────────────────

# HSV COLOUR RANGES (one per fret lane)
FRET_HSV = [
    ((37,56,65), (82,255,255)),  # 0 Green
    ((0,39,46), (18,255,188)),  # 1 Red
    ((27,39,46), (52,255,188)),  # 2 Yellow
    ((92,39,46), (109,255,188)),  # 3 Blue
    ((11,24,48), (26,255,221)),  # 4 Orange
]

# Minimum fraction of hit-zone pixels that must be lit for a HIT confirmation
HIT_PIXEL_THRESHOLD = 0.08

# Action smoothing - prevent rapid toggle
ACTION_MIN_HOLD_FRAMES = 2  # Minimum frames to hold a key before release
