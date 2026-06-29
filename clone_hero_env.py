import mss
import cv2
import numpy as np
import pydirectinput
import gymnasium as gym
from gymnasium import spaces
import time
import re
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'
pydirectinput.PAUSE = 0

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

HIGHWAY_REGION = {
    "top":    300,
    "left":   430,
    "width":  500,
    "height": 400
}

# Strikebar — where notes must be hit
STRIKEBAR_Y = 340

# Sustain detection — how far above the strikebar a long note extends
# If color is detected in this band AND at strikebar → it's a sustain
SUSTAIN_LOOKAHEAD_Y = 200  # pixels above strikebar to check for sustain tail

HEALTH_REGION = {
    "top":    50,
    "left":   200,
    "width":  400,
    "height": 20
}

STATS_REGION = {
    "top":    193,
    "left":   828,
    "width":  195,
    "height": 147,
}

LANE_COLORS_BGR = {
    0: (3,   128,  3),    # Green
    1: (0,   0,    164),  # Red
    2: (57,  183,  183),  # Yellow
    3: (153, 82,   0),    # Blue
    4: (0,   135,  192),  # Orange
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
REPEAT_INTERVAL = 300  # 5 minutes

# How often to run OCR stats (every N steps) — expensive so not every step
STATS_READ_INTERVAL = 200

# ─────────────────────────────────────────────────────────────
# ACTION SPACE  (11 discrete actions)
#
#   0–4  → strum lane 0–4  (tap: press + strum + release)
#   5–9  → hold  lane 0–4  (sustain: keyDown only, no strum)
#   10   → release all     (end sustain / do nothing)
#
# The agent learns: use action 0-4 to hit a note head,
# use action 5-9 to keep holding during a sustain tail,
# use action 10 to release when the sustain ends.
# ─────────────────────────────────────────────────────────────

class CloneHeroEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug

        self.action_space      = spaces.Discrete(11)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

        self.sct = mss.MSS()  # uppercase — avoids deprecation + random freeze bug

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
        self.stats_monitor = {
            "top":    mon["top"]  + STATS_REGION["top"],
            "left":   mon["left"] + STATS_REGION["left"],
            "width":  STATS_REGION["width"],
            "height": STATS_REGION["height"],
        }

        self.lane_width        = HIGHWAY_REGION["width"] // 5
        self.prev_frame_bgr    = None
        self.current_frame_bgr = None
        self._first_reset      = True
        self._last_repeat_time = time.time()
        self._held_keys        = set()   # tracks which keys are currently held down
        self._step_count       = 0
        self._prev_ghosts      = 0       # ghost note count from last OCR read
        self._last_stats       = {}      # most recent parsed stats

    # ─────────────────────────────────────────────────────────
    # CORE GYM METHODS
    # ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._release_all_keys()
        self.prev_frame_bgr    = None
        self.current_frame_bgr = None
        self._step_count       = 0
        self._prev_ghosts      = 0
        self._last_stats       = {}

        if self._first_reset:
            self._first_reset = False
            time.sleep(0.5)
        else:
            self._restart_song()

        self._last_repeat_time = time.time()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self._maybe_repeat_song()
        self._execute_action(action)

        obs    = self._get_obs()
        reward = self._compute_reward(action)

        # read OCR stats every STATS_READ_INTERVAL steps
        self._step_count += 1
        if self._step_count % STATS_READ_INTERVAL == 0:
            self._last_stats = self._read_stats()
            if self.debug and self._last_stats:
                print(f"  [stats] {self._last_stats}")

        # ghost note penalty from OCR
        ghosts = self._last_stats.get("ghosts")
        if ghosts is not None:
            ghost_delta = ghosts - self._prev_ghosts
            if ghost_delta > 0:
                reward -= ghost_delta * 0.3
                if self.debug:
                    print(f"  GHOST x{ghost_delta} penalty")
            self._prev_ghosts = ghosts

        song_failed = self._is_song_failed()
        terminated  = song_failed
        truncated   = False

        if song_failed:
            reward -= 10.0
            self._release_all_keys()

        if self.current_frame_bgr is not None:
            self.prev_frame_bgr = self.current_frame_bgr.copy()

        info = {"stats": self._last_stats}
        return obs, reward, terminated, truncated, info

    def close(self):
        self._release_all_keys()
        self.sct.close()

    # ─────────────────────────────────────────────────────────
    # ACTION EXECUTION
    # ─────────────────────────────────────────────────────────

    def _execute_action(self, action):
        if action <= 4:
            # STRUM action — tap the note (press + strum + release)
            # release any held keys first so we don't corrupt the strum
            self._release_all_keys()
            key = LANE_KEYS[action]
            pydirectinput.keyDown(key)
            pydirectinput.press(STRUM_KEY)
            pydirectinput.keyUp(key)

        elif action <= 9:
            # HOLD action — sustain, keep key down, no strum
            lane = action - 5
            key  = LANE_KEYS[lane]
            if key not in self._held_keys:
                pydirectinput.keyDown(key)
                self._held_keys.add(key)

        else:
            # action 10 — release everything
            self._release_all_keys()

    def _release_all_keys(self):
        for key in LANE_KEYS.values():
            pydirectinput.keyUp(key)
        self._held_keys.clear()

    # ─────────────────────────────────────────────────────────
    # SONG MANAGEMENT
    # ─────────────────────────────────────────────────────────

    def _restart_song(self):
        if self.debug:
            print("  [reset] restarting song via 'h'...")
        pydirectinput.press(REPEAT_KEY)
        time.sleep(2.0)
        if self.debug:
            print("  [reset] song restarted")

    def _maybe_repeat_song(self):
        now = time.time()
        if now - self._last_repeat_time >= REPEAT_INTERVAL:
            if self.debug:
                print(f"  [repeat] {REPEAT_INTERVAL}s elapsed — pressing 'h'")
            self._release_all_keys()
            pydirectinput.press(REPEAT_KEY)
            time.sleep(2.0)
            self._last_repeat_time = now
            self.prev_frame_bgr    = None

    # ─────────────────────────────────────────────────────────
    # OBSERVATION
    # ─────────────────────────────────────────────────────────

    def _get_obs(self):
        raw       = self.sct.grab(self.highway_monitor)
        frame_bgr = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
        self.current_frame_bgr = frame_bgr

        gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        if self.debug:
            cv2.imwrite("debug_obs.png",   resized)
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
            at_strikebar = self._note_present_in_lane(lane_idx, self.prev_frame_bgr)

            if not at_strikebar:
                continue

            is_sustain = self._is_sustain(lane_idx, self.prev_frame_bgr)

            if is_sustain:
                # sustain note — agent should be holding (action 5-9)
                hold_action = lane_idx + 5
                if action == hold_action:
                    reward += 0.5   # smaller reward per frame for holding
                    if self.debug:
                        print(f"  HOLD lane {lane_idx} ✓")
                else:
                    reward -= 0.3   # not holding during sustain
                    if self.debug:
                        print(f"  HOLD lane {lane_idx} ✗ (agent chose {action})")
            else:
                # regular note — agent should strum (action 0-4)
                if action == lane_idx:
                    reward += 1.0
                    if self.debug:
                        print(f"  HIT  lane {lane_idx} ✓")
                else:
                    reward -= 0.5
                    if self.debug:
                        print(f"  MISS lane {lane_idx} ✗ (agent chose {action})")

        return reward

    def _note_present_in_lane(self, lane_idx, frame_bgr):
        """Returns True if note color is in the strikebar zone of this lane."""
        x_start = lane_idx * self.lane_width
        x_end   = x_start + self.lane_width
        y_start = STRIKEBAR_Y
        y_end   = HIGHWAY_REGION["height"]

        lane_zone  = frame_bgr[y_start:y_end, x_start:x_end]
        if lane_zone.size == 0:
            return False

        target_bgr = LANE_COLORS_BGR[lane_idx]
        lower = np.array([max(0,   c - COLOR_TOLERANCE) for c in target_bgr], dtype=np.uint8)
        upper = np.array([min(255, c + COLOR_TOLERANCE) for c in target_bgr], dtype=np.uint8)
        mask  = cv2.inRange(lane_zone, lower, upper)

        ratio = np.count_nonzero(mask) / mask.size
        result = ratio > 0.05

        if self.debug:
            mean_color = lane_zone.mean(axis=(0, 1))
            lane_names = ["Green", "Red", "Yellow", "Blue", "Orange"]
            print(
                f"  [strikebar] lane {lane_idx} ({lane_names[lane_idx]})  "
                f"zone=({x_start}:{x_end}, {y_start}:{y_end})  "
                f"target=({target_bgr[0]},{target_bgr[1]},{target_bgr[2]})  "
                f"mean=({int(mean_color[0])},{int(mean_color[1])},{int(mean_color[2])})  "
                f"match_ratio={ratio:.3f}  "
                f"→ {'NOTE' if result else 'none'}"
            )

        return result

    def _is_sustain(self, lane_idx, frame_bgr):
        """
        Returns True if this note is a sustain (long note).
        A sustain has color continuously extending above the strikebar.
        We check a band from SUSTAIN_LOOKAHEAD_Y to STRIKEBAR_Y.
        If a significant portion of that band matches the note color → sustain.
        """
        x_start = lane_idx * self.lane_width
        x_end   = x_start + self.lane_width
        y_start = max(0, STRIKEBAR_Y - SUSTAIN_LOOKAHEAD_Y)
        y_end   = STRIKEBAR_Y

        sustain_zone = frame_bgr[y_start:y_end, x_start:x_end]
        if sustain_zone.size == 0:
            return False

        target_bgr = LANE_COLORS_BGR[lane_idx]
        lower = np.array([max(0,   c - COLOR_TOLERANCE) for c in target_bgr], dtype=np.uint8)
        upper = np.array([min(255, c + COLOR_TOLERANCE) for c in target_bgr], dtype=np.uint8)
        mask  = cv2.inRange(sustain_zone, lower, upper)

        # count rows that have any matching pixels
        rows_with_color = np.any(mask > 0, axis=1).sum()
        total_rows      = sustain_zone.shape[0]

        # if more than 30% of rows above strikebar have this color → sustain
        return bool(rows_with_color / total_rows > 0.30)

    # ─────────────────────────────────────────────────────────
    # OCR STATS
    # ─────────────────────────────────────────────────────────

    def _read_stats(self):
        """Capture and OCR the stats panel. Returns dict, empty dict on failure."""
        try:
            raw  = self.sct.grab(self.stats_monitor)
            img  = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            scale = 3
            large = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale),
                               interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            text  = pytesseract.image_to_string(thresh, config='--psm 6')
            clean = text.replace(" ", "").replace("\n", " ")

            stats = {}

            m = re.search(r"[Hh]it[Nn]otes[\W_]*(\d+)/(\d+)", clean)
            if m:
                stats["hit_notes"]   = int(m.group(1))
                stats["total_notes"] = int(m.group(2))

            m = re.search(r"[Hh]it[Pp]ercent[\W_]*([\d.]+)%", clean)
            if m:
                stats["hit_percent"] = float(m.group(1))

            m = re.search(r"[Ff]ret[Gg]hosts[\W_]*(\d+)", clean)
            if m:
                stats["ghosts"] = int(m.group(1))

            m = re.search(r"(\d[\d.,]*)[Nn][Pp][Ss]", clean)
            if m:
                raw_nps = m.group(1).replace(",", ".")
                parts   = raw_nps.split(".")
                try:
                    stats["nps"] = float(parts[0] + ("." + parts[1] if len(parts) > 1 else ""))
                except ValueError:
                    pass

            return stats

        except Exception as e:
            if self.debug:
                print(f"  [stats] OCR failed: {e}")
            return {}

    # ─────────────────────────────────────────────────────────
    # SONG FAILURE DETECTION
    # ─────────────────────────────────────────────────────────

    def _is_song_failed(self):
        raw  = self.sct.grab(self.health_monitor)
        gray = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2GRAY)
        return float(gray.mean()) < 15.0