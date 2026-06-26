import mss
import cv2
import numpy as np
import pydirectinput
import gymnasium as gym
from gymnasium import spaces
import time

pydirectinput.PAUSE = 0

# ─────────────────────────────────────────────────────────────
# CONSTANTS — calibrate these with calibrate.py first
# ─────────────────────────────────────────────────────────────

HIGHWAY_REGION = {
    "top":    300,
    "left":   430,
    "width":  500,
    "height": 400
}

STRIKEBAR_Y = 340  # Y within the cropped highway where notes get hit

HEALTH_REGION = {
    "top":    50,
    "left":   200,
    "width":  400,
    "height": 20
}

# Note colors in BGR — update these from calibrate.py output
LANE_COLORS_BGR = {
    0: (3,   128,  3),    # Green
    1: (0,   0,    164),  # Red
    2: (57,   183,  183),  # Yellow
    3: (153, 82,    0),    # Blue
    4: (0, 135,  192),    # Orange
}
COLOR_TOLERANCE = 60

# Keys — match your Clone Hero key bindings
LANE_KEYS = {
    0: 'q',   # Green
    1: 's',   # Red
    2: 'j',   # Yellow
    3: 'k',   # Blue
    4: 'l',   # Orange
}
STRUM_KEY = 'down'

MAX_STEPS = 3000

# ─────────────────────────────────────────────────────────────
# ACTION SPACE
#   0 → Green  + strum
#   1 → Red    + strum
#   2 → Yellow + strum
#   3 → Blue   + strum
#   4 → Orange + strum
#   5 → do nothing
# ─────────────────────────────────────────────────────────────

class CloneHeroEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

        self.sct = mss.mss()  # lowercase — mss.MSS() leaks resources

        try:
            mon = self.sct.monitors[2]
        except IndexError:
            mon = self.sct.monitors[1]

        self.highway_monitor = {
            "top":    mon["top"]  + HIGHWAY_REGION["top"],
            "left":   mon["left"] + HIGHWAY_REGION["left"],
            "width":  HIGHWAY_REGION["width"],
            "height": HIGHWAY_REGION["height"],
        }

        self.health_monitor = {
            "top":    mon["top"]  + HEALTH_REGION["top"],
            "left":   mon["left"] + HEALTH_REGION["left"],
            "width":  HEALTH_REGION["width"],
            "height": HEALTH_REGION["height"],
        }

        self.lane_width = HIGHWAY_REGION["width"] // 5
        self.prev_frame_bgr = None
        self.current_frame_bgr = None
        self.steps = 0

    # ─────────────────────────────────────────────────────────
    # CORE GYM METHODS
    # ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Always release everything on reset — no stuck keys
        self._release_all_keys()

        self.prev_frame_bgr = None
        self.current_frame_bgr = None
        self.steps = 0

        time.sleep(0.3)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Execute and immediately release — no held state
        self._execute_action(action)

        obs = self._get_obs()
        reward = self._compute_reward(action)

        self.steps += 1

        song_failed = self._is_song_failed()
        terminated = song_failed or self.steps >= MAX_STEPS
        truncated = False

        if song_failed:
            reward -= 10.0

        if terminated:
            self._release_all_keys()

        if self.current_frame_bgr is not None:
            self.prev_frame_bgr = self.current_frame_bgr.copy()

        return obs, reward, terminated, truncated, {}

    def close(self):
        self._release_all_keys()
        self.sct.close()

    # ─────────────────────────────────────────────────────────
    # ACTION EXECUTION
    # ─────────────────────────────────────────────────────────

    def _execute_action(self, action):
        """
        For a fret action: press key down, strum, release key.
        The key is held only during the strum — then released immediately.
        Action 5 = do nothing.
        """
        if action == 5:
            return

        key = LANE_KEYS[action]

        pydirectinput.keyDown(key)
        pydirectinput.press(STRUM_KEY)  # strum while fret is held
        pydirectinput.keyUp(key)        # release immediately after strum

    def _release_all_keys(self):
        """Force-release every fret key. Called on reset and termination."""
        for key in LANE_KEYS.values():
            pydirectinput.keyUp(key)

    # ─────────────────────────────────────────────────────────
    # OBSERVATION
    # ─────────────────────────────────────────────────────────

    def _get_obs(self):
        raw = self.sct.grab(self.highway_monitor)
        frame_bgr = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
        self.current_frame_bgr = frame_bgr

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        if self.debug:
            cv2.imwrite("debug_obs.png", resized)
            cv2.imwrite("debug_color.png", frame_bgr)

        return np.expand_dims(resized, axis=-1)

    # ─────────────────────────────────────────────────────────
    # REWARD
    # ─────────────────────────────────────────────────────────

    def _compute_reward(self, action):
        reward = -0.01  # time penalty

        if self.prev_frame_bgr is None:
            return reward

        for lane_idx in range(5):
            note_present = self._note_present_in_lane(lane_idx, self.prev_frame_bgr)

            if note_present:
                if action == lane_idx:
                    reward += 1.0
                    if self.debug:
                        print(f"  HIT  lane {lane_idx}")
                else:
                    reward -= 0.5
                    if self.debug:
                        print(f"  MISS lane {lane_idx} — agent chose {action}")

        return reward

    def _note_present_in_lane(self, lane_idx, frame_bgr):
        """Returns True if the correct note color is visible in this lane's strikebar zone."""
        x_start = lane_idx * self.lane_width
        x_end   = x_start + self.lane_width
        y_start = STRIKEBAR_Y
        y_end   = HIGHWAY_REGION["height"]

        lane_zone = frame_bgr[y_start:y_end, x_start:x_end]

        if lane_zone.size == 0:
            return False

        target_bgr = LANE_COLORS_BGR[lane_idx]
        lower = np.array([max(0,   c - COLOR_TOLERANCE) for c in target_bgr], dtype=np.uint8)
        upper = np.array([min(255, c + COLOR_TOLERANCE) for c in target_bgr], dtype=np.uint8)
        mask = cv2.inRange(lane_zone, lower, upper)

        match_ratio = np.count_nonzero(mask) / mask.size
        return bool(match_ratio > 0.05)

    # ─────────────────────────────────────────────────────────
    # SONG FAILURE DETECTION
    # ─────────────────────────────────────────────────────────

    def _is_song_failed(self):
        """Health bar goes dark when empty — song is over."""
        raw = self.sct.grab(self.health_monitor)
        gray = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2GRAY)
        return float(gray.mean()) < 15.0