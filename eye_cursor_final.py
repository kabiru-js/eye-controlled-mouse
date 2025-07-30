# eye_cursor_final_v2.py (Corrected and Final)

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import keyboard
import time
import json
import sys

def load_or_create_config():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            print("Configuration loaded successfully from config.json.")
            return config
    except FileNotFoundError:
        print("Config file not found. Creating a default 'config.json'.")
        default_config = {
          "SMOOTHING_FACTOR": 0.7,
          "HORIZONTAL_SENSITIVITY": [0.2, 0.8],
          "VERTICAL_SENSITIVITY": [0.3, 0.5],
          "BLINK_EAR_THRESHOLD": 0.2,
          "CLICK_DURATION_SECONDS": 0.25,
          "DOUBLE_CLICK_INTERVAL_SECONDS": 0.5
        }
        with open('config.json', 'w') as f:
            json.dump(default_config, f, indent=2)
        print("Default config file created. Please run the script again.")
        sys.exit()
    except json.JSONDecodeError:
        print("Error: 'config.json' is malformed. Please check its syntax.")
        sys.exit()
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        sys.exit()

def main():
    config = load_or_create_config()
    SCREEN_W, SCREEN_H = pyautogui.size()
    
    SMOOTHING_FACTOR = config['SMOOTHING_FACTOR']
    HORIZONTAL_SENSITIVITY = config['HORIZONTAL_SENSITIVITY']
    VERTICAL_SENSITIVITY = config['VERTICAL_SENSITIVITY']
    BLINK_EAR_THRESHOLD = config['BLINK_EAR_THRESHOLD']
    CLICK_DURATION_SECONDS = config['CLICK_DURATION_SECONDS']
    DOUBLE_CLICK_INTERVAL_SECONDS = config['DOUBLE_CLICK_INTERVAL_SECONDS']
    
    print("\n--- Eye-Controlled Cursor (Final Version) ---")
    print("INFO: Application is now running in the background.")
    print("\n--- CONTROLS ---")
    print("- Look to move the cursor.")
    print("- Close BOTH eyes for a moment to LEFT CLICK.")
    print("- Wink your RIGHT eye for a moment to RIGHT CLICK.")
    print("- Perform two quick blinks for a DOUBLE CLICK.")
    print("- Press SPACEBAR to Pause or Resume cursor control.")
    print("- Press 'q' to quit the application.")
    print("----------------")
    
    is_paused = False
    pause_key_pressed = False
    left_eye_closed_start, right_eye_closed_start = None, None
    is_left_clicking, is_right_clicking = False, False
    last_click_time = 0
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Could not open any webcam. Please check connections.")
            return

    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    smooth_x, smooth_y = 0, 0
    print("\nTracking started. Cursor is ACTIVE.")
    
    LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

    while True:
        if keyboard.is_pressed('space'):
            if not pause_key_pressed:
                is_paused = not is_paused
                status = "PAUSED" if is_paused else "ACTIVE"
                print(f"Cursor control is now {status}.")
                pause_key_pressed = True
                time.sleep(0.2)
        else:
            pause_key_pressed = False

        if not is_paused:
            success, frame = cap.read()
            if not success: continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                left_pupil = face_landmarks.landmark[473]
                right_pupil = face_landmarks.landmark[468]
                avg_pupil_x_norm = (left_pupil.x + right_pupil.x) / 2
                avg_pupil_y_norm = (left_pupil.y + right_pupil.y) / 2
                
                screen_x = np.interp(avg_pupil_x_norm, HORIZONTAL_SENSITIVITY, [0, SCREEN_W])
                screen_y = np.interp(avg_pupil_y_norm, VERTICAL_SENSITIVITY, [0, SCREEN_H])
                screen_x = SCREEN_W - screen_x
                
                # --- THIS IS THE CRUCIAL FIX ---
                screen_x = np.clip(screen_x, 1, SCREEN_W - 1)
                screen_y = np.clip(screen_y, 1, SCREEN_H - 1)
                
                smooth_x = smooth_x * SMOOTHING_FACTOR + screen_x * (1 - SMOOTHING_FACTOR)
                smooth_y = smooth_y * SMOOTHING_FACTOR + screen_y * (1 - SMOOTHING_FACTOR)
                pyautogui.moveTo(smooth_x, smooth_y, duration=0)

                left_ear = get_eye_aspect_ratio(LEFT_EYE_LANDMARKS, face_landmarks)
                right_ear = get_eye_aspect_ratio(RIGHT_EYE_LANDMARKS, face_landmarks)

                if left_ear < BLINK_EAR_THRESHOLD:
                    if left_eye_closed_start is None: left_eye_closed_start = time.time()
                else:
                    left_eye_closed_start = None; is_left_clicking = False

                if right_ear < BLINK_EAR_THRESHOLD:
                    if right_eye_closed_start is None: right_eye_closed_start = time.time()
                else:
                    right_eye_closed_start = None; is_right_clicking = False

                now = time.time()
                if left_eye_closed_start and right_eye_closed_start and not is_left_clicking:
                    duration = min(now - left_eye_closed_start, now - right_eye_closed_start)
                    if duration >= CLICK_DURATION_SECONDS:
                        if (now - last_click_time) < DOUBLE_CLICK_INTERVAL_SECONDS:
                            pyautogui.doubleClick(); print("DOUBLE CLICK!")
                        else:
                            pyautogui.click(); print("LEFT CLICK!")
                        last_click_time = now; is_left_clicking = True
                elif right_eye_closed_start and not left_eye_closed_start and not is_right_clicking:
                    if left_ear > (BLINK_EAR_THRESHOLD + 0.1):
                        duration = now - right_eye_closed_start
                        if duration >= CLICK_DURATION_SECONDS:
                            pyautogui.rightClick(); print("RIGHT CLICK!")
                            is_right_clicking = True
        else:
            time.sleep(0.1)

        if keyboard.is_pressed('q'): break
    
    face_mesh.close()
    cap.release()
    print("Application closed.")

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_eye_aspect_ratio(eye_landmarks, face_landmarks):
    p1=[face_landmarks.landmark[l].x for l in eye_landmarks]
    p2=[face_landmarks.landmark[l].y for l in eye_landmarks]
    vertical_dist = euclidean_distance((p1[1], p2[1]), (p1[5], p2[5])) + euclidean_distance((p1[2], p2[2]), (p1[4], p2[4]))
    horizontal_dist = euclidean_distance((p1[0], p2[0]), (p1[3], p2[3]))
    if horizontal_dist == 0: return 0
    return vertical_dist / (2.0 * horizontal_dist)

if __name__ == '__main__':
    main()