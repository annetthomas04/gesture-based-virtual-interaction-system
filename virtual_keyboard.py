import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller, Key
import time
import math
import pyautogui

class VirtualKeyboard:
    HANDS_LABELS = {"Left": "Left", "Right": "Right"}

    # Colors (BGR)
    C_DARK   = (18, 18, 28)
    C_PANEL  = (28, 32, 48)
    C_ACCENT = (0, 200, 160)
    C_GREEN  = (80, 220, 120)
    C_YELLOW = (60, 220, 220)
    C_ORANGE = (60, 160, 255)
    C_WHITE  = (240, 240, 240)
    C_GRAY   = (100, 100, 120)
    C_PURPLE = (200, 100, 220)
    C_RED    = (80, 80, 220)

    KEY_NORMAL = (45, 48, 68)
    KEY_HOVER  = (65, 80, 110)
    KEY_ACTIVE = (0, 160, 120)

    def __init__(self, mp_hands, hands, mp_draw, window_width, window_height):
        self.mp_hands = mp_hands
        self.hands    = hands
        self.mp_draw  = mp_draw

        self.window_width  = window_width
        self.window_height = window_height

        self.keyboard = Controller()

        self.keys = {
            'normal': [
                ['Esc','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12'],
                ['`','1','2','3','4','5','6','7','8','9','0','-','=','Backspace'],
                ['Tab','q','w','e','r','t','y','u','i','o','p','[',']','\\'],
                ['Caps','a','s','d','f','g','h','j','k','l',';',"'",'Enter'],
                ['Shift','z','x','c','v','b','n','m',',','.','/', 'Shift'],
                ['Ctrl','Win','Alt','Space','Alt','Ctrl']
            ],
            'shift': [
                ['Esc','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12'],
                ['~','!','@','#','$','%','^','&','*','(',')', '_','+','Backspace'],
                ['Tab','Q','W','E','R','T','Y','U','I','O','P','{','}','|'],
                ['Caps','A','S','D','F','G','H','J','K','L',':','"','Enter'],
                ['Shift','Z','X','C','V','B','N','M','<','>','?','Shift'],
                ['Ctrl','Win','Alt','Space','Alt','Ctrl']
            ],
            'ctrl': [
                ['Esc','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12'],
                ['`','1','2','3','4','5','6','7','8','9','0','-','=','Backspace'],
                ['Tab','Q','W','E','R','T','Y','U','I','O','P','[',']','\\'],
                ['Caps','A','S','D','F','G','H','J','K','L',';',"'",'Enter'],
                ['Shift','Z','X','C','V','B','N','M',',','.','/', 'Shift'],
                ['Ctrl','Win','Alt','Space','Alt','Ctrl']
            ]
        }

        self.special_keys = {
            'Space': Key.space, 'Tab': Key.tab, 'Enter': Key.enter,
            'Backspace': Key.backspace, 'Esc': Key.esc, 'Win': Key.cmd,
            'F1': Key.f1, 'F2': Key.f2, 'F3': Key.f3, 'F4': Key.f4,
            'F5': Key.f5, 'F6': Key.f6, 'F7': Key.f7, 'F8': Key.f8,
            'F9': Key.f9, 'F10': Key.f10, 'F11': Key.f11, 'F12': Key.f12,
        }

        self.shift_pressed = False
        self.caps_lock     = False
        self.ctrl_pressed  = False
        self.alt_pressed   = False

        self.button_width  = 40
        self.button_height = 40
        self.button_margin = 4

        self.key_widths = {
            'Backspace': 80, 'Tab': 60, 'Caps': 80, 'Enter': 80,
            'Shift': 100,    'Ctrl': 60, 'Alt': 60, 'Space': 240, 'Win': 60
        }

        self.clicked      = False
        self.click_cooldown = 0.22
        self.last_click_time = 0
        self.prev_clicked = False

        # Hovered key this frame (for visual feedback)
        self.hovered_key  = None
        self.last_pressed_key = None
        self.last_pressed_time = 0

        # ── Accuracy tracking ──────────────────────────────────────────────────
        # Records (intended_key, detected_key, timestamp)
        # "intended" = what the user declared via set_expected_key()
        # "detected" = what was actually registered
        self.accuracy_log        = []    # list of (intended, detected, correct)
        self.total_presses       = 0
        self.correct_presses     = 0
        self.expected_key        = None  # set externally or via UI toggle
        self.tracking_accuracy   = False # toggle on/off

        # Per-key counts: {key: {'attempts': n, 'hits': n}}
        self.per_key_stats = {}

        # Finger role legend for left hand
        self.finger_roles = {
            'INDEX':  'Hover / Point',
            'THUMB':  'Pinch = Press',
            'MIDDLE': '—',
            'RING':   '—',
            'PINKY':  '—',
        }
        self.finger_colors = {
            'INDEX':  (80,  220, 120),
            'THUMB':  (60,  200, 255),
            'MIDDLE': (120, 120, 120),
            'RING':   (120, 120, 120),
            'PINKY':  (120, 120, 120),
        }

    # ──────────────────────────────────────── key geometry helpers
    def get_key_width(self, key):
        return self.key_widths.get(key, self.button_width)

    def _iter_keys(self, layout):
        """Yield (key, x, y, w, h) for every key in current layout."""
        ks = 20
        cy = ks
        for row in self.keys[layout]:
            cx = ks
            for key in row:
                w = self.get_key_width(key)
                yield key, cx, cy, w, self.button_height
                cx += w + self.button_margin
            cy += self.button_height + self.button_margin

    def _current_layout(self):
        if self.shift_pressed: return 'shift'
        if self.ctrl_pressed:  return 'ctrl'
        return 'normal'

    # ──────────────────────────────────────── drawing
    def draw_keyboard(self, img):
        layout = self._current_layout()
        hovered = self.hovered_key

        for key, cx, cy, w, h in self._iter_keys(layout):
            # Background
            if key == hovered:
                bg = self.KEY_HOVER
            else:
                bg = self.KEY_NORMAL
            cv2.rectangle(img, (cx, cy), (cx+w, cy+h), bg, -1)

            # Border color
            border = self.C_WHITE
            if key in ('Shift',) and self.shift_pressed:
                border = self.C_GREEN
            elif key == 'Caps' and self.caps_lock:
                border = self.C_GREEN
            elif key == 'Ctrl' and self.ctrl_pressed:
                border = self.C_GREEN
            elif self.ctrl_pressed and key in ('C','V','X','Z','Y'):
                border = self.C_YELLOW
            elif key == hovered:
                border = self.C_ACCENT

            # Flash pressed key
            if (self.last_pressed_key == key and
                    time.time() - self.last_pressed_time < 0.18):
                cv2.rectangle(img, (cx, cy), (cx+w, cy+h), self.KEY_ACTIVE, -1)
                border = self.C_GREEN

            cv2.rectangle(img, (cx, cy), (cx+w, cy+h), border, 1)

            # Label text
            font_scale = 0.75 if len(key) == 1 else 0.45
            (tw, th), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx = cx + (w - tw) // 2
            ty = cy + (h + th) // 2
            col = self.C_YELLOW if (self.ctrl_pressed and key in ('C','V','X','Z','Y')) else self.C_WHITE
            cv2.putText(img, key, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, col, 1, cv2.LINE_AA)

    def _draw_legend(self, img):
        """Right-side legend for left hand roles."""
        entries = [
            ("LEFT HAND — KEYBOARD", self.C_ACCENT),
            ("", None),
            ("INDEX finger", self.finger_colors['INDEX']),
            ("  → Hover over key", self.C_WHITE),
            ("THUMB pinch INDEX", self.finger_colors['THUMB']),
            ("  → Press hovered key", self.C_WHITE),
            ("", None),
            ("SHIFT  → Uppercase/Sym", self.C_GREEN),
            ("CAPS   → Caps Lock", self.C_GREEN),
            ("CTRL   → Shortcuts", self.C_YELLOW),
        ]
        x0 = img.shape[1] - 215
        y0 = 12
        h_total = len(entries) * 18 + 14
        cv2.rectangle(img, (x0-8, y0-4), (img.shape[1]-4, y0+h_total),
                      self.C_PANEL, -1)
        cv2.rectangle(img, (x0-8, y0-4), (img.shape[1]-4, y0+h_total),
                      self.C_ACCENT, 1)
        for i, item in enumerate(entries):
            if item[1] is None:
                continue
            txt, col = item
            cv2.putText(img, txt, (x0, y0 + i*18 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1, cv2.LINE_AA)

    def _draw_accuracy_panel(self, img):
        """Bottom strip showing live accuracy stats."""
        if not self.tracking_accuracy:
            return
        acc = (self.correct_presses / self.total_presses * 100
               if self.total_presses > 0 else 0.0)
        panel_y = img.shape[0] - 36
        cv2.rectangle(img, (0, panel_y), (img.shape[1], img.shape[0]),
                      self.C_PANEL, -1)
        cv2.rectangle(img, (0, panel_y), (img.shape[1], img.shape[0]),
                      self.C_ACCENT, 1)

        stats = (f"ACCURACY MODE   "
                 f"Presses: {self.total_presses}   "
                 f"Correct: {self.correct_presses}   "
                 f"Accuracy: {acc:.1f}%   "
                 f"Expected: [{self.expected_key or 'none'}]")
        cv2.putText(img, stats, (10, panel_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.C_GREEN, 1, cv2.LINE_AA)

    def _draw_status_bar(self, img, is_clicked):
        """Bottom status: current mode + pinch state."""
        mode = self._current_layout().upper()
        modifiers = []
        if self.shift_pressed: modifiers.append("SHIFT")
        if self.caps_lock:     modifiers.append("CAPS")
        if self.ctrl_pressed:  modifiers.append("CTRL")
        if self.alt_pressed:   modifiers.append("ALT")

        mod_str = " + ".join(modifiers) if modifiers else "NORMAL"
        pinch   = "  ● PINCH DETECTED" if is_clicked else ""
        bar_y   = img.shape[0] - (62 if self.tracking_accuracy else 26)

        cv2.rectangle(img, (0, bar_y), (img.shape[1], bar_y + 22),
                      (35, 38, 55), -1)
        cv2.putText(img, f"Mode: {mod_str}{pinch}",
                    (10, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_WHITE, 1, cv2.LINE_AA)
        if self.last_pressed_key and time.time() - self.last_pressed_time < 0.6:
            cv2.putText(img, f"Last: {self.last_pressed_key}",
                        (img.shape[1] - 120, bar_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_GREEN, 1, cv2.LINE_AA)

    def _draw_finger_labels(self, img, hand_landmarks):
        h, w, _ = img.shape
        names    = ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']
        tip_ids  = [4, 8, 12, 16, 20]
        for name, tip_id in zip(names, tip_ids):
            lm  = hand_landmarks.landmark[tip_id]
            px  = int(lm.x * w)
            py  = int(lm.y * h)
            col = self.finger_colors.get(name, self.C_GRAY)
            cv2.circle(img, (px, py), 7, col, -1)
            cv2.circle(img, (px, py), 9, self.C_WHITE, 1)
            # Label
            (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
            bx, by = px + 11, py - 8
            cv2.rectangle(img, (bx-2, by-10), (bx+tw+2, by+2), self.C_DARK, -1)
            cv2.putText(img, name, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, col, 1, cv2.LINE_AA)

    def _draw_pinch_indicator(self, img, hand_landmarks):
        """Draw a line between thumb and index tip showing distance."""
        h, w, _ = img.shape
        tx = int(hand_landmarks.landmark[4].x * w)
        ty = int(hand_landmarks.landmark[4].y * h)
        ix = int(hand_landmarks.landmark[8].x * w)
        iy = int(hand_landmarks.landmark[8].y * h)
        dist = math.sqrt((tx-ix)**2 + (ty-iy)**2)
        # Normalize to window scale
        norm_dist = math.sqrt(
            (hand_landmarks.landmark[4].x - hand_landmarks.landmark[8].x)**2 +
            (hand_landmarks.landmark[4].y - hand_landmarks.landmark[8].y)**2
        )
        col = self.C_GREEN if norm_dist < 0.05 else self.C_ORANGE
        cv2.line(img, (tx, ty), (ix, iy), col, 2, cv2.LINE_AA)
        mid = ((tx+ix)//2, (ty+iy)//2)
        cv2.putText(img, f"{norm_dist:.3f}", (mid[0]+4, mid[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1, cv2.LINE_AA)

    # ──────────────────────────────────────── hit detection
    def get_clicked_key(self, finger_pos):
        layout = self._current_layout()
        x, y   = finger_pos
        for key, cx, cy, w, h in self._iter_keys(layout):
            if cx < x < cx + w and cy < y < cy + h:
                return key
        return None

    # ──────────────────────────────────────── key press logic
    def handle_key_press(self, key):
        if key is None:
            return

        # Record for accuracy
        self._record_press(key)

        self.last_pressed_key  = key
        self.last_pressed_time = time.time()

        if key == 'Shift':
            self.shift_pressed = not self.shift_pressed
        elif key == 'Caps':
            self.caps_lock = not self.caps_lock
        elif key == 'Ctrl':
            self.ctrl_pressed = not self.ctrl_pressed
        elif key == 'Alt':
            self.alt_pressed = not self.alt_pressed
        elif key in self.special_keys:
            sk = self.special_keys[key]
            self.keyboard.press(sk)
            self.keyboard.release(sk)
        else:
            char = key
            if len(key) == 1:
                char = key.upper() if (self.caps_lock != self.shift_pressed) else key.lower()

            if self.ctrl_pressed:
                mapping = {'c': 'c', 'v': 'v', 'x': 'x', 'z': 'z', 'y': 'y'}
                if key.lower() in mapping:
                    pyautogui.hotkey('ctrl', mapping[key.lower()])
                return

            self.keyboard.press(char)
            self.keyboard.release(char)

            if self.shift_pressed and key != 'Shift':
                self.shift_pressed = False

    # ──────────────────────────────────────── accuracy tracking
    def _record_press(self, detected_key):
        """Log this keypress against the expected key (if tracking)."""
        if not self.tracking_accuracy:
            return
        self.total_presses += 1

        # Per-key tracking regardless of expected
        k = detected_key
        if k not in self.per_key_stats:
            self.per_key_stats[k] = {'attempts': 0, 'hits': 0}
        self.per_key_stats[k]['attempts'] += 1

        if self.expected_key is not None:
            correct = (detected_key == self.expected_key)
            if correct:
                self.correct_presses += 1
                self.per_key_stats[k]['hits'] += 1
            self.accuracy_log.append({
                'expected': self.expected_key,
                'detected': detected_key,
                'correct':  correct,
                'time':     time.time(),
            })

    def set_expected_key(self, key):
        """Call this before each test key to declare ground truth."""
        self.expected_key = key

    def start_accuracy_tracking(self):
        self.tracking_accuracy   = True
        self.total_presses       = 0
        self.correct_presses     = 0
        self.accuracy_log        = []
        self.per_key_stats       = {}
        print("[Accuracy] Tracking started.")

    def stop_accuracy_tracking(self):
        self.tracking_accuracy = False
        self.print_accuracy_report()

    def print_accuracy_report(self):
        if self.total_presses == 0:
            print("[Accuracy] No presses recorded.")
            return
        acc = self.correct_presses / self.total_presses * 100
        print("\n" + "="*52)
        print("  KEYBOARD ACCURACY REPORT")
        print("="*52)
        print(f"  Total presses   : {self.total_presses}")
        print(f"  Correct presses : {self.correct_presses}")
        print(f"  Accuracy        : {acc:.1f}%")
        print(f"  Error rate      : {100-acc:.1f}%")
        print()
        if self.accuracy_log:
            errors = [e for e in self.accuracy_log if not e['correct']]
            print(f"  Errors ({len(errors)}):")
            for e in errors:
                print(f"    Expected '{e['expected']}' → Got '{e['detected']}'")
        print()
        if self.per_key_stats:
            print("  Per-key hit rate:")
            for k, stats in sorted(self.per_key_stats.items()):
                rate = stats['hits'] / stats['attempts'] * 100 if stats['attempts'] else 0
                bar  = "█" * int(rate // 10)
                print(f"    [{k:>12s}]  {rate:5.1f}%  {bar}")
        print("="*52 + "\n")

    # ──────────────────────────────────────── pinch detection
    def detect_click(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]
        distance  = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2
        )
        return distance < 0.05

    # ──────────────────────────────────────── main handler
    def handle_hand_gestures(self, hands_processing_results, left_hand_index, img):
        if (hands_processing_results is None
                or hands_processing_results.multi_hand_landmarks is None
                or img is None):
            return

        if left_hand_index is None:
            cv2.putText(img, "LEFT HAND", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_ACCENT, 1, cv2.LINE_AA)
            cv2.putText(img, "not detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_GRAY, 1, cv2.LINE_AA)
            self.hovered_key = None
            self.draw_keyboard(img)
            self._draw_legend(img)
            self._draw_accuracy_panel(img)
            return

        hand_landmarks = hands_processing_results.multi_hand_landmarks[left_hand_index]

        # Index fingertip → keyboard canvas position
        index_tip = hand_landmarks.landmark[8]
        finger_x  = int(index_tip.x * self.window_width)
        finger_y  = int(index_tip.y * self.window_height)
        finger_x  = max(0, min(finger_x, self.window_width  - 1))
        finger_y  = max(0, min(finger_y, self.window_height - 1))

        # Update hovered key
        self.hovered_key = self.get_clicked_key((finger_x, finger_y))

        # Draw keyboard (hover highlight applied inside)
        self.draw_keyboard(img)

        # Draw hand skeleton + labels
        self.mp_draw.draw_landmarks(
            img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(60, 60, 80), thickness=1, circle_radius=0),
            self.mp_draw.DrawingSpec(color=(80, 80, 120), thickness=1)
        )
        self._draw_finger_labels(img, hand_landmarks)
        self._draw_pinch_indicator(img, hand_landmarks)

        # Cursor dot on index tip
        cv2.circle(img, (finger_x, finger_y), 6, self.C_ACCENT, -1)
        cv2.circle(img, (finger_x, finger_y), 8, self.C_WHITE,  1)

        # Pinch detection + cooldown
        is_clicked   = self.detect_click(hand_landmarks)
        current_time = time.time()

        if (is_clicked and not self.prev_clicked
                and current_time - self.last_click_time > self.click_cooldown):
            clicked_key = self.get_clicked_key((finger_x, finger_y))
            if clicked_key:
                self.handle_key_press(clicked_key)
                self.last_click_time = current_time

        # Draw overlays
        self._draw_legend(img)
        self._draw_status_bar(img, is_clicked)
        self._draw_accuracy_panel(img)

        self.prev_clicked = is_clicked
