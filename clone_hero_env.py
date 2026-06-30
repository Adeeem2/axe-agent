import mss
import cv2
import numpy as np
import pydirectinput
import gymnasium as gym
from gymnasium import spaces
import pytesseract
import time
import re
from collections import deque
import os
import sys

from config import (
    HIGHWAY_REGION, HEALTH_REGION, STATS_REGION,
    LANE_KEYS, STRUM_KEY, REPEAT_KEY, REPEAT_INTERVAL,
    STATS_READ_INTERVAL,
    # New config
    REWARD_SCORE_SCALE, REWARD_GHOST_PENALTY, REWARD_FAIL_PENALTY,
    REWARD_SURVIVAL_BONUS, REWARD_HIT_BONUS, REWARD_COMBO_BONUS,
    OBS_GRAYSCALE, OBS_RESIZE, ACTION_MIN_HOLD_FRAMES,
    REQUIRE_ADMIN,
)

# ─────────────────────────────────────────────────────────────
# ADMIN CHECK (Windows)
# ─────────────────────────────────────────────────────────────
def check_admin():
    """Check if running as administrator on Windows."""
    if REQUIRE_ADMIN and os.name == 'nt':
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            pass
    return True

pydirectinput.PAUSE = 0

# ─────────────────────────────────────────────────────────────
# ACTION SPACE  MultiBinary(6)
#
#   action[0:5] → fret mask: 0 = release, 1 = press
#                 lanes 0–4 = Green, Red, Yellow, Blue, Orange
#   action[5]   → strum:     0 = no, 1 = yes
#
# Examples:
#   [1,0,0,0,0,1] → press Green + strum                 (single note)
#   [1,0,1,0,0,1] → press Green + Yellow + strum        (chord)
#   [1,0,0,0,0,0] → hold Green  (sustain, no strum)
#   [0,0,0,0,0,0] → release all / do nothing
# ─────────────────────────────────────────────────────────────

class CloneHeroEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, debug=False, curriculum_stage=0):
        super().__init__()

        self.debug = debug
        self.curriculum_stage = curriculum_stage

        # Check admin rights
        if not check_admin():
            print("⚠️  WARNING: Not running as Administrator! pydirectinput may not work.")
            print("   Run your terminal/IDE as Administrator.")

        self.action_space      = spaces.MultiBinary(6)
        # Observation: 4 stacked frames, RGB or grayscale
        n_channels = 1 if OBS_GRAYSCALE else 3
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(OBS_RESIZE[1], OBS_RESIZE[0], n_channels * 4), dtype=np.uint8
        )

        self.sct = mss.MSS()

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

        # Frame stack
        self.frame_stack       = deque(maxlen=4)

        # Action tracking with minimum hold frames
        self._prev_action      = np.zeros(6, dtype=int)
        self._action_hold_frames = np.zeros(6, dtype=int)  # Track how long each key held

        # Reward tracking
        self._prev_score       = None
        self._prev_ghosts      = 0
        self._prev_hit_percent = 0.0
        self._prev_combo       = 0
        self._last_stats       = {}
        self._first_reset      = True
        self._held_keys        = set()
        self._step_count       = 0
        self._episode_steps    = 0
        self._total_reward     = 0.0

        # Song management - DISABLED automatic repeat to prevent "stops clicking"
        # self._last_repeat_time = time.time()
        # self._song_repeat_enabled = False  # Set True only if you want auto-repeat

    # ─────────────────────────────────────────────────────────
    # CORE GYM METHODS
    # ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._release_all_keys()
        self.frame_stack.clear()
        self._prev_action = np.zeros(6, dtype=int)
        self._action_hold_frames = np.zeros(6, dtype=int)
        self._prev_score  = None
        self._prev_ghosts = 0
        self._prev_hit_percent = 0.0
        self._prev_combo = 0
        self._last_stats  = {}
        self._step_count  = 0
        self._episode_steps = 0
        self._total_reward = 0.0

        if self._first_reset:
            self._first_reset = False
            time.sleep(0.5)
        else:
            self._restart_song()

        # Fill frame stack with initial frames
        rgb = self._capture_rgb()
        for _ in range(4):
            self.frame_stack.append(rgb)

        return self._build_obs(), {}

    def step(self, action):
        self._episode_steps += 1

        # Execute action with minimum hold frames
        self._execute_action(action)

        # Capture post-action frame and build stacked observation
        obs = self._get_obs()

        # Reward calculation
        reward = 0.0
        self._step_count += 1

        # Dense survival bonus
        reward += REWARD_SURVIVAL_BONUS

        # Read stats periodically
        if self._step_count % STATS_READ_INTERVAL == 0:
            stats = self._read_stats()
            self._last_stats = stats

            # Score increase → positive reward
            score = stats.get("score")
            if score is not None and self._prev_score is not None:
                delta = score - self._prev_score
                if delta > 0:
                    reward += delta * REWARD_SCORE_SCALE
                    if self.debug:
                        print(f"  SCORE +{delta} → reward +{delta * REWARD_SCORE_SCALE:.2f}")

            # Ghost notes → penalty
            ghosts = stats.get("ghosts")
            if ghosts is not None:
                ghost_delta = ghosts - self._prev_ghosts
                if ghost_delta > 0:
                    reward -= ghost_delta * REWARD_GHOST_PENALTY
                    if self.debug:
                        print(f"  GHOST x{ghost_delta} penalty")

            # Hit percent improvement bonus
            hit_percent = stats.get("hit_percent")
            if hit_percent is not None and self._prev_hit_percent > 0:
                if hit_percent > self._prev_hit_percent:
                    reward += (hit_percent - self._prev_hit_percent) * REWARD_HIT_BONUS
            if hit_percent is not None:
                self._prev_hit_percent = hit_percent

            # Combo bonus (if available in stats)
            combo = stats.get("combo", 0)
            if combo > self._prev_combo:
                reward += (combo - self._prev_combo) * REWARD_COMBO_BONUS
                self._prev_combo = combo

            if score is not None:
                self._prev_score = score
            if ghosts is not None:
                self._prev_ghosts = ghosts

        song_failed = self._is_song_failed()
        terminated  = song_failed
        truncated   = False

        if song_failed:
            reward += REWARD_FAIL_PENALTY
            self._release_all_keys()

        self._total_reward += reward

        info = {
            "stats": self._last_stats,
            "episode_steps": self._episode_steps,
            "total_reward": self._total_reward,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self._release_all_keys()
        self.sct.close()

    # ─────────────────────────────────────────────────────────
    # ACTION EXECUTION  (diff-based with minimum hold frames)
    # ─────────────────────────────────────────────────────────

    def _execute_action(self, action):
        """Press/release only keys whose state changed from previous action.
        Enforces minimum hold frames to prevent rapid toggling."""
        for i in range(5):
            key = LANE_KEYS[i]
            prev = self._prev_action[i]
            curr = action[i]

            if curr and not prev:
                # Press new key
                pydirectinput.keyDown(key)
                self._held_keys.add(key)
                self._action_hold_frames[i] = 1
            elif curr and prev:
                # Continue holding - increment hold counter
                self._action_hold_frames[i] += 1
            elif not curr and prev:
                # Release key only if minimum hold frames met
                if self._action_hold_frames[i] >= ACTION_MIN_HOLD_FRAMES:
                    pydirectinput.keyUp(key)
                    self._held_keys.discard(key)
                    self._action_hold_frames[i] = 0
                else:
                    # Not held long enough - keep pressed (override action)
                    action[i] = 1
                    self._action_hold_frames[i] += 1
            else:
                # Not pressed, not previously pressed
                self._action_hold_frames[i] = 0

        # Strum is momentary - press on action[5] == 1
        if action[5]:
            pydirectinput.press(STRUM_KEY)

        self._prev_action = action.copy()

    def _release_all_keys(self):
        for key in list(self._held_keys):
            pydirectinput.keyUp(key)
        self._held_keys.clear()
        self._action_hold_frames = np.zeros(6, dtype=int)

    # ─────────────────────────────────────────────────────────
    # SONG MANAGEMENT
    # ─────────────────────────────────────────────────────────

    def _restart_song(self):
        if self.debug:
            print("  [reset] restarting song via 'h'...")
        self._release_all_keys()
        time.sleep(0.2)
        pydirectinput.press(REPEAT_KEY)
        time.sleep(2.0)  # Wait for song to load
        if self.debug:
            print("  [reset] song restarted")

    # DISABLED: Automatic song repeat every 5 minutes was causing "stops clicking"
    # def _maybe_repeat_song(self):
    #     now = time.time()
    #     if now - self._last_repeat_time >= REPEAT_INTERVAL:
    #         if self.debug:
    #             print(f"  [repeat] {REPEAT_INTERVAL}s elapsed — pressing 'h'")
    #         self._release_all_keys()
    #         pydirectinput.press(REPEAT_KEY)
    #         time.sleep(2.0)
    #         self._last_repeat_time = now
    #         # Song restarted — reset tracking and re-fill frame stack
    #         self._prev_score = None
    #         self._prev_ghosts = 0
    #         self._prev_hit_percent = 0.0
    #         self._prev_combo = 0
    #         self.frame_stack.clear()
    #         rgb = self._capture_rgb()
    #         for _ in range(4):
    #             self.frame_stack.append(rgb)

    # ─────────────────────────────────────────────────────────
    # OBSERVATION  — 84×84 RGB × 4 stacked frames
    # ─────────────────────────────────────────────────────────

    def _capture_rgb(self):
        """Capture highway region, return resized RGB or grayscale array."""
        raw       = self.sct.grab(self.highway_monitor)
        frame_bgr = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)

        if OBS_GRAYSCALE:
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame = frame[:, :, np.newaxis]  # Add channel dim
        else:
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(frame, OBS_RESIZE, interpolation=cv2.INTER_AREA)

        if self.debug:
            if OBS_GRAYSCALE:
                cv2.imwrite("debug_obs.png", resized.squeeze())
            else:
                cv2.imwrite("debug_color.png", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

        return resized

    def _build_obs(self):
        """Stack 4 frames into (H, W, C*4)."""
        return np.concatenate(list(self.frame_stack), axis=-1)

    def _get_obs(self):
        """Capture frame, push to stack, return stacked observation."""
        frame = self._capture_rgb()
        self.frame_stack.append(frame)
        return self._build_obs()

    # ─────────────────────────────────────────────────────────
    # OCR STATS  (including score)
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

            # Score — primary reward signal
            m = re.search(r"[Ss]core[:\s]*([\d,]+)", clean)
            if m:
                stats["score"] = int(m.group(1).replace(",", ""))

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

            # Try to extract combo if present
            m = re.search(r"[Cc]ombo[:\s]*(\d+)", clean)
            if m:
                stats["combo"] = int(m.group(1))

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
