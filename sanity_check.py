import mss
import cv2
import numpy as np
import os

def capture_and_test_crops():
    with mss.MSS() as sct:
        # Get Monitor 2
        try:
            mon2 = sct.monitors[2]
        except IndexError:
            print("Monitor 2 not found, using Monitor 1")
            mon2 = sct.monitors[1]
        
        print(f"Capturing from: {mon2}")
        
        # 1. Full Screenshot of Monitor 2
        full_screenshot = np.array(sct.grab(mon2))
        cv2.imwrite("debug_full_mon2.png", full_screenshot)
        print("Saved debug_full_mon2.png")
        
        # 2. Test Highway Crop
        # Given 1360x768, let's try a centered crop if it's full screen
        # Or let's use some safer values for 768 height
        highway_top = 300
        highway_height = 400 # 300 + 400 = 700 < 768
        highway_left = 430
        highway_width = 500
        
        monitor_highway = {
            "top": mon2["top"] + highway_top,
            "left": mon2["left"] + highway_left,
            "width": highway_width,
            "height": highway_height
        }
        
        highway_img = np.array(sct.grab(monitor_highway))
        cv2.imwrite("debug_highway.png", highway_img)
        print(f"Saved debug_highway.png (top={highway_top}, left={highway_left}, w={highway_width}, h={highway_height})")
        
        # 3. Crop each note (lane)
        # Assuming 5 lanes
        lane_width = highway_width // 5
        for i in range(5):
            lane_crop = highway_img[:, i*lane_width : (i+1)*lane_width]
            cv2.imwrite(f"debug_lane_{i}.png", lane_crop)
            print(f"Saved debug_lane_{i}.png")
            
            # Optionally further crop the "hit zone" (flame area) at the bottom
            hit_zone_height = 100
            hit_zone = lane_crop[highway_height - hit_zone_height : , :]
            cv2.imwrite(f"debug_flame_{i}.png", hit_zone)
            print(f"Saved debug_flame_{i}.png")

if __name__ == "__main__":
    capture_and_test_crops()
