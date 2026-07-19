# Eye-Controlled Cursor System

A real-time computer vision system that enables hands-free mouse control using eye movements and blink detection via a standard webcam.

---

## 🧠 Problem

Traditional input devices (mouse, trackpad) are not accessible to everyone and can be inefficient in certain environments.

This project explores **human-computer interaction through gaze tracking**, allowing users to control a cursor using only their eyes.

---

## ⚙️ How It Works

The system processes webcam input in real time and performs:

1. **Face & Eye Detection**
   - Detects facial landmarks using computer vision
   - Isolates eye regions

2. **Gaze Estimation**
   - Maps eye position to screen coordinates
   - Translates gaze into cursor movement

3. **Blink Detection**
   - Left click → both eyes closed briefly  
   - Right click → right eye wink  
   - Double click → rapid blinking  

4. **Cursor Control**
   - Sends OS-level mouse events based on interpreted signals

---

## 🧱 System Architecture

```text
Webcam Input
      ↓
Eye Detection (Computer Vision)
      ↓
Gaze Mapping
      ↓
Blink Detection
      ↓
Cursor Controller (OS events)
