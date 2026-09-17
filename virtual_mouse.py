import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time

class VirtualMouse:
    HANDS_LABELS = {"Left": "Left", "Right": "Right"}

    # Finger landmark indices for labels
    FINGER_TIPS = {
        'THUMB':  4,
        'INDEX':  8,
        'MIDDLE': 12,
        'RING':   16,
        'PINKY':  20,
    }

    # What each finger does for the mouse
    FINGER_ROLES = {
        'THUMB':  'Click modifier',
        'INDEX':  'Move / Click',
        'MIDDLE': 'Right-click',
        'RING':   'Hold/Drag',
        'PINKY':  '—',
    }

    # Colors (BGR)
    C_GREEN   = (80, 220, 120)
    C_CYAN    = (220, 220, 60)
    C_ORANGE  = (60, 160, 255)
    C_RED     = (60, 60, 220)
    C_WHITE   = (240, 240, 240)
    C_GRAY    = (120, 120, 120)
    C_DARK    = (18, 18, 28)
    C_PANEL   = (28, 32, 48)
    C_ACCENT  = (0, 200, 160)

    FINGER_COLORS = {
        'THUMB':  (60,  200, 255),
        'INDEX':  (80,  220, 120),
        'MIDDLE': (60,  160, 255),
        'RING':   (180, 100, 255),
        'PINKY':  (120, 120, 120),
    }

    def __init__(self, mp_hands, hands, mp_draw, window_width, window_height):
        self.mp_hands = mp_hands
        self.hands    = hands
        self.mp_draw  = mp_draw

        self.screen_width, self.screen_height = pyautogui.size()
        self.frame_reduction = 50
        self.smoothening     = 2
        self.prev_x, self.prev_y = 0, 0

        self.window_width  = window_width
        self.window_height = window_height

        self.last_click_time        = 0
        self.double_click_threshold = 0.3

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE    = 0.01

        self.prev_left_click  = False
        self.prev_right_click = False
        self.is_holding       = False

        self.current_gesture  = "No Hand"
        self.gesture_history  = []   # for smoothing display

    # ------------------------------------------------------------------ helpers
    def calculate_distance(self, p1, p2):
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

    def get_finger_positions(self, hand_landmarks, img_shape):
        h, w, _ = img_shape
        landmarks = {
            'thumb':  (int(hand_landmarks.landmark[4].x  * w), int(hand_landmarks.landmark[4].y  * h)),
            'index':  (int(hand_landmarks.landmark[8].x  * w), int(hand_landmarks.landmark[8].y  * h)),
            'middle': (int(hand_landmarks.landmark[12].x * w), int(hand_landmarks.landmark[12].y * h)),
            'ring':   (int(hand_landmarks.landmark[16].x * w), int(hand_landmarks.landmark[16].y * h)),
        }
        is_finger_up = {
            'thumb':  hand_landmarks.landmark[4].x  > hand_landmarks.landmark[3].x,
            'index':  hand_landmarks.landmark[8].y  < hand_landmarks.landmark[6].y,
            'middle': hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,
            'ring':   hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,
        }
        return landmarks, is_finger_up

    def detect_gestures(self, hand_landmarks, img_shape):
        landmarks, is_finger_up = self.get_finger_positions(hand_landmarks, img_shape)

        left_click  = (    is_finger_up['thumb'] and     is_finger_up['index']
                       and not is_finger_up['middle'] and not is_finger_up['ring'])
        right_click = (not is_finger_up['thumb'] and     is_finger_up['index']
                       and is_finger_up['middle']    and not is_finger_up['ring'])
        click_hold  = (not is_finger_up['thumb'] and     is_finger_up['index']
                       and is_finger_up['middle']    and     is_finger_up['ring'])

        return left_click, right_click, click_hold

    def move_mouse(self, finger_pos):
        frame_x = np.interp(finger_pos[0],
                            (self.frame_reduction, 640 - self.frame_reduction),
                            (0, self.screen_width))
        frame_y = np.interp(finger_pos[1],
                            (self.frame_reduction, 480 - self.frame_reduction),
                            (0, self.screen_height))
        current_x = self.prev_x + (frame_x - self.prev_x) / self.smoothening
        current_y = self.prev_y + (frame_y - self.prev_y) / self.smoothening
        pyautogui.moveTo(current_x, current_y)
        self.prev_x, self.prev_y = current_x, current_y

    # ------------------------------------------------------------------ drawing
    def _draw_panel_bg(self, img):
        """Dark semi-transparent left panel strip."""
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]),
                      self.C_DARK, -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    def _draw_finger_labels(self, img, hand_landmarks):
        h, w, _ = img.shape
        finger_order = ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']
        tip_ids = [4, 8, 12, 16, 20]
        is_up = {
            'THUMB':  hand_landmarks.landmark[4].x  > hand_landmarks.landmark[3].x,
            'INDEX':  hand_landmarks.landmark[8].y  < hand_landmarks.landmark[6].y,
            'MIDDLE': hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,
            'RING':   hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,
            'PINKY':  hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y,
        }

        for name, tip_id in zip(finger_order, tip_ids):
            lm  = hand_landmarks.landmark[tip_id]
            px  = int(lm.x * w)
            py  = int(lm.y * h)
            col = self.FINGER_COLORS[name]
            state = "UP" if is_up[name] else "down"

            # Dot on fingertip
            cv2.circle(img, (px, py), 7, col, -1)
            cv2.circle(img, (px, py), 9, self.C_WHITE, 1)

            # Label bubble
            label = f"{name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            bx, by = px + 12, py - 10
            cv2.rectangle(img, (bx - 3, by - th - 3), (bx + tw + 3, by + 3),
                          self.C_DARK, -1)
            cv2.putText(img, label, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, col, 1, cv2.LINE_AA)

            # State tag
            state_col = self.C_GREEN if is_up[name] else self.C_GRAY
            cv2.putText(img, state, (bx, by + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, state_col, 1, cv2.LINE_AA)

    def _draw_gesture_badge(self, img, gesture_name, left_click, right_click, click_hold):
        """Gesture name badge at top of camera panel."""
        badge_col = self.C_GRAY
        if left_click:
            badge_col = self.C_GREEN
        elif right_click:
            badge_col = self.C_ORANGE
        elif click_hold:
            badge_col = self.C_RED

        text = f"  {gesture_name}  "
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bx, by = 10, 10
        cv2.rectangle(img, (bx, by), (bx + tw + 6, by + th + 10), badge_col, -1)
        cv2.putText(img, text, (bx + 3, by + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.C_DARK, 1, cv2.LINE_AA)

    def _draw_legend(self, img):
        """Side legend: gesture → action mapping."""
        entries = [
            ("RIGHT HAND — MOUSE", None, self.C_ACCENT),
            ("", None, None),
            ("INDEX up", "Move cursor", self.C_GREEN),
            ("THUMB + INDEX", "Left click", self.C_GREEN),
            ("INDEX + MIDDLE", "Right click", self.C_ORANGE),
            ("INDEX+MIDDLE+RING", "Hold / Drag", self.C_RED),
            ("Double gesture", "Double click", self.C_CYAN),
        ]
        x0 = img.shape[1] - 210
        y0 = 12
        # Panel background
        cv2.rectangle(img, (x0 - 8, y0 - 4),
                      (img.shape[1] - 4, y0 + len(entries) * 22 + 6),
                      self.C_PANEL, -1)
        cv2.rectangle(img, (x0 - 8, y0 - 4),
                      (img.shape[1] - 4, y0 + len(entries) * 22 + 6),
                      self.C_ACCENT, 1)

        for i, (gesture, action, col) in enumerate(entries):
            y = y0 + i * 22 + 16
            if col is None:
                continue
            if action is None:
                # Section header
                cv2.putText(img, gesture, (x0, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, gesture, (x0, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1, cv2.LINE_AA)
                cv2.putText(img, f"→ {action}", (x0, y + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, self.C_WHITE, 1, cv2.LINE_AA)

    def _draw_cursor_crosshair(self, img, pos):
        x, y = pos
        col = self.C_ACCENT
        cv2.line(img, (x - 12, y), (x + 12, y), col, 1, cv2.LINE_AA)
        cv2.line(img, (x, y - 12), (x, y + 12), col, 1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 5, col, -1)

    # ------------------------------------------------------------------ main
    def handle_hand_gestures(self, hands_processing_results, right_hand_index, img):
        if (hands_processing_results is None
                or hands_processing_results.multi_hand_landmarks is None
                or img is None):
            return

        if right_hand_index is None:
            self.current_gesture = "No Right Hand"
            # Draw minimal no-hand overlay
            cv2.putText(img, "RIGHT HAND", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_ACCENT, 1, cv2.LINE_AA)
            cv2.putText(img, "not detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_GRAY, 1, cv2.LINE_AA)
            self._draw_legend(img)
            if self.is_holding:
                pyautogui.mouseUp()
                self.is_holding = False
            return

        hand_landmarks = hands_processing_results.multi_hand_landmarks[right_hand_index]
        landmarks, _ = self.get_finger_positions(hand_landmarks, img.shape)
        left_click, right_click, click_hold = self.detect_gestures(hand_landmarks, img.shape)

        # Determine gesture label
        if left_click:
            self.current_gesture = "LEFT CLICK"
        elif right_click:
            self.current_gesture = "RIGHT CLICK"
        elif click_hold:
            self.current_gesture = "HOLD / DRAG"
        else:
            self.current_gesture = "MOVING"

        # Draw custom hand skeleton with labels
        self._draw_finger_labels(img, hand_landmarks)
        # Draw connections manually so they sit under our dots
        self.mp_draw.draw_landmarks(
            img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(60, 60, 80), thickness=1, circle_radius=0),
            self.mp_draw.DrawingSpec(color=(80, 80, 120), thickness=1)
        )
        # Re-draw our dots on top
        self._draw_finger_labels(img, hand_landmarks)

        # Cursor crosshair on index tip
        self._draw_cursor_crosshair(img, landmarks['index'])

        # Legend and badge
        self._draw_legend(img)
        self._draw_gesture_badge(img, self.current_gesture, left_click, right_click, click_hold)

        # Move mouse
        self.move_mouse(landmarks['index'])

        # Left click / double click
        if left_click and not self.prev_left_click:
            current_time = time.time()
            if current_time - self.last_click_time < self.double_click_threshold:
                pyautogui.doubleClick()
                cv2.circle(img, landmarks['index'], 18, self.C_CYAN, 2)
            else:
                pyautogui.click()
                cv2.circle(img, landmarks['index'], 14, self.C_GREEN, 2)
            self.last_click_time = current_time

        # Right click
        if right_click and not self.prev_right_click:
            pyautogui.rightClick()
            cv2.circle(img, landmarks['index'], 14, self.C_ORANGE, 2)

        # Hold
        if click_hold and not self.is_holding:
            pyautogui.mouseDown()
            self.is_holding = True
        elif not click_hold and self.is_holding:
            pyautogui.mouseUp()
            self.is_holding = False

        self.prev_left_click  = left_click
        self.prev_right_click = right_click
