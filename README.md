# Gesture-Based Virtual Interaction System

A real-time, webcam-only touchless input system that replaces a physical mouse and keyboard using hand gesture recognition. No specialist hardware or wearables required, just a standard webcam and a Python environment.

Built as a Final Year Project at the University of West London, School of Computing and Engineering.

---

## Demo

> Right hand controls the mouse. Left hand types on the virtual keyboard.

---

## How It Works

MediaPipe Hands extracts 21 three-dimensional landmarks per hand from each webcam frame. A rule-based finite state machine reads the up/down state of each finger and maps combinations to OS-level input events; injected system-wide via PyAutoGUI and pynput, indistinguishable from a physical device.

```
Webcam frame
    → OpenCV (flip, colour convert)
    → MediaPipe (21 landmarks × hand)
    → Gesture FSM (finger state combinations)
    → PyAutoGUI / pynput (OS input injection)
    → Any active application
```

---

## Features

- **Virtual Mouse** — right hand
  - Index finger pointing → cursor movement with low-pass smoothening
  - Thumb + index up → left click (double-click on repeat within 300ms)
  - Index + middle up → right click
  - Index + middle + ring up → click and hold / drag

- **Virtual Keyboard** — left hand
  - Index finger hover → key selection
  - Thumb-index pinch → key activation
  - Full QWERTY layout with Shift, Caps Lock, and Ctrl state support
  - Ctrl shortcuts: C, V, X, Z, Y

- **Dual-hand simultaneous operation** — mouse and keyboard active in the same frame loop

- **Live UI overlays**
  - Finger labels and state indicators (UP / down) on the camera feed
  - Gesture badge showing current active gesture
  - Pinch distance visualiser between thumb and index tip
  - Key hover highlight and press flash on the keyboard canvas
  - Status bar showing active modifier keys and last registered key

- **Keystroke accuracy evaluation**
  - Built-in accuracy tracking module logging intended vs. detected keypresses
  - Per-key hit rate, overall accuracy %, and error log
  - Automated 26-letter test sequence triggered from the keyboard

---

## Performance

| Metric | Value |
|---|---|
| Frame rate (single hand) | 25–30 FPS |
| Frame rate (dual hand) | 18–25 FPS |
| End-to-end latency | 30–60 ms |
| Gesture recognition accuracy | ~85% macro-average |
| Keyboard hit accuracy (letters) | ~78–82% |
| Keyboard hit accuracy (large keys) | ~95%+ |

---

## Installation

**Requirements:** Python 3.9–3.11, a standard webcam

```bash

# 1. Clone repository and create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

---

## Usage

| Control | Action |
|---|---|
| Right hand — index up | Move cursor |
| Right hand — thumb + index | Left click |
| Right hand — index + middle | Right click |
| Right hand — index + middle + ring | Click and hold |
| Left hand — index hover | Select key |
| Left hand — thumb pinch index | Press key |
| Press `Shift` on virtual keyboard | Uppercase / symbols |
| Press `Ctrl` on virtual keyboard | Shortcut mode (C V X Z Y) |
| Press `Q` (physical) | Quit |
| Press `T` (physical) | Start accuracy test |
| Press `S` (physical) | Stop test and print report |

---

## Project Structure

```
gesture-virtual-interaction/
│
├── main.py                 # Orchestrator — webcam loop, hand resolution, display
├── virtual_mouse.py        # Right-hand gesture logic, mouse output, UI overlays
├── virtual_keyboard.py     # Left-hand gesture logic, keyboard rendering, accuracy tracking
├── requirements.txt
└── README.md
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `mediapipe` | Hand landmark detection (21 points per hand) |
| `opencv-python` | Frame capture, image processing, UI rendering |
| `pyautogui` | Mouse control, hotkey injection |
| `pynput` | Keyboard press/release event injection |
| `numpy` | Coordinate interpolation and smoothening |

---

## Known Limitations

- No scroll gesture support in the current version
- Accuracy degrades under poor or inconsistent lighting

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

---
"# gesture-virtual-interaction-system" 
"# gesture-virtual-interaction-system" 
"# gesture-virtual-interaction-system" 
