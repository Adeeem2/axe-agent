import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

HIGHWAY_REGION = {
    "top":    300,
    "left":   430,
    "width":  500,
    "height": 400,
}

STRIKEBAR_Y = 340

SUSTAIN_LOOKAHEAD_Y = 200

HEALTH_REGION = {
    "top":    50,
    "left":   200,
    "width":  400,
    "height": 20,
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
