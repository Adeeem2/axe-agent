import time
import os
import mss
import mss.tools
import pydirectinput

# Faster inputs for games
pydirectinput.PAUSE = 0

GREEN_FRET_KEY = 'q'
RED_FRET_KEY = 's'
YELLOW_FRET_KEY = 'j'
BLUE_FRET_KEY = 'k'
ORANGE_FRET_KEY = 'l'
strum = 'up'


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def take_screenshot(path):
    with mss.MSS() as sct:
        monitor = sct.monitors[2] if len(sct.monitors) > 2 else sct.monitors[1]
        sct_img = sct.grab(monitor)
        mss.tools.to_png(sct_img.rgb, (sct_img.width, sct_img.height), output=path)


def main():
    for i in range(2, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    before_path = os.path.join(PROJECT_DIR, "screenshot_before.png")
    after_path = os.path.join(PROJECT_DIR, "screenshot_after.png")

    print("Taking initial screenshot...")
    take_screenshot(before_path)
    time.sleep(0.5)

    try:
        pydirectinput.keyDown(GREEN_FRET_KEY)
        pydirectinput.keyDown(strum)
        time.sleep(0.5)
    finally:
        pydirectinput.keyUp(strum)
        pydirectinput.keyUp(GREEN_FRET_KEY)

    time.sleep(0.2)

    print("Taking screenshot after keypress...")
    take_screenshot(after_path)

    print('\nDone. Look at the two screenshots in the project folder:')
    print(f" - {before_path}")
    print(f" - {after_path}")
    print("If Clone Hero was focused, you should see a visible reaction (strum/fret lighting, score change, etc.).")


if __name__ == '__main__':
    main()

