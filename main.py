import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from performance_logger import PerformanceLogger
from virtual_mouse    import VirtualMouse
from virtual_keyboard import VirtualKeyboard

class MouseAndKeyboard:
    HANDS_LABELS = {"Left": "Left", "Right": "Right"}

    C_DARK   = ( 18,  18,  28)
    C_PANEL  = ( 28,  32,  48)
    C_ACCENT = (  0, 200, 160)
    C_WHITE  = (240, 240, 240)
    C_GRAY   = (100, 100, 120)
    C_GREEN  = ( 80, 220, 120)
    C_ORANGE = ( 60, 160, 255)
    C_RED    = ( 80,  80, 220)

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.window_width  = 1000
        self.window_height = 420      # slightly taller for header room

        self.mouse    = VirtualMouse(
            self.mp_hands, self.hands, self.mp_draw,
            self.window_width, self.window_height)
        self.keyboard = VirtualKeyboard(
            self.mp_hands, self.hands, self.mp_draw,
            self.window_width, self.window_height)

        # ── accuracy test sequence (optional) ─────────────────────────────────
        # When 't' is pressed the system cycles through these keys and prompts
        # the user to press each one; results are printed on 's'.
        self._test_sequence = list("abcdefghijklmnopqrstuvwxyz")
        self._test_idx       = 0
        self._test_running   = False

    def _draw_header(self, img):
        """Top header bar across the full combined image."""
        HEADER_H = 36
        cv2.rectangle(img, (0, 0), (img.shape[1], HEADER_H), self.C_PANEL, -1)
        cv2.line(img, (0, HEADER_H), (img.shape[1], HEADER_H), self.C_ACCENT, 1)

        cv2.putText(img, "GESTURE-BASED VIRTUAL INTERACTION SYSTEM",
                    (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    self.C_ACCENT, 1, cv2.LINE_AA)

        # Right side: key hints
        hints = "[Q] Quit   [T] Start accuracy test   [S] Stop & report   [A] Set expected key"
        (tw, _), _ = cv2.getTextSize(hints, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
        cv2.putText(img, hints, (img.shape[1] - tw - 10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, self.C_GRAY, 1, cv2.LINE_AA)

    def _draw_section_label(self, img, text, x, y, col):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(img, (x, y), (x + tw + 14, y + th + 8), self.C_PANEL, -1)
        cv2.rectangle(img, (x, y), (x + tw + 14, y + th + 8), col, 1)
        cv2.putText(img, text, (x + 7, y + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    def _draw_divider(self, img, x):
        """Vertical divider between camera and keyboard panel."""
        cv2.line(img, (x, 36), (x, img.shape[0]), self.C_PANEL, 8)
        cv2.line(img, (x, 36), (x, img.shape[0]), self.C_ACCENT, 1)

    # ─────────────────────────────────────────────────────── accuracy overlay
    def _draw_test_overlay(self, img):
        if not self._test_running:
            return
        key = self._test_sequence[self._test_idx]
        text = f"ACCURACY TEST  Press key: [ {key.upper()} ]"
        prog = f"{self._test_idx+1} / {len(self._test_sequence)}"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cx = (img.shape[1] - tw) // 2
        cv2.rectangle(img, (cx - 10, img.shape[0] - 60),
                      (cx + tw + 10, img.shape[0] - 22),
                      (0, 0, 0), -1)
        cv2.rectangle(img, (cx - 10, img.shape[0] - 60),
                      (cx + tw + 10, img.shape[0] - 22),
                      self.C_ACCENT, 1)
        cv2.putText(img, text, (cx, img.shape[0] - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.C_GREEN, 1, cv2.LINE_AA)
        cv2.putText(img, prog,
                    (img.shape[1] - 80, img.shape[0] - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_GRAY, 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────── accuracy test
    def _start_test(self):
        self.keyboard.start_accuracy_tracking()
        self._test_idx     = 0
        self._test_running = True
        self.keyboard.set_expected_key(self._test_sequence[0])
        print(f"[Test] Please press key on virtual keyboard: "
              f"'{self._test_sequence[0].upper()}'")

    def _advance_test(self):
        """Called whenever a key is pressed during test mode."""
        if not self._test_running:
            return
        self._test_idx += 1
        if self._test_idx >= len(self._test_sequence):
            self._test_running = False
            self.keyboard.stop_accuracy_tracking()
            print("[Test] Complete.")
        else:
            nk = self._test_sequence[self._test_idx]
            self.keyboard.set_expected_key(nk)
            print(f"[Test] Press key: '{nk.upper()}'")

    def start(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.window_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_height)
        logger = PerformanceLogger()

        screen_w = pyautogui.size()[0]
        screen_h = pyautogui.size()[1]
        win_x    = (screen_w - self.window_width) // 2
        win_y    = screen_h - self.window_height - 40

        WINDOW_NAME = "Gesture-Based Virtual Interaction System"
        cv2.namedWindow(WINDOW_NAME,
                        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(WINDOW_NAME, win_x, win_y)

        HEADER_H   = 36
        prev_kb_presses = 0

        try:
            while True:
                logger.frame_start()
                success, camera_img = cap.read(); logger.stage('frame_acq')
                if not success:
                    continue

                camera_img     = cv2.flip(camera_img, 1)
                rgb_camera_img = cv2.cvtColor(camera_img, cv2.COLOR_BGR2RGB)
                results        = self.hands.process(rgb_camera_img); logger.stage('mediapipe')

                # ── dark background for keyboard panel ────────────────────────────
                kb_canvas = np.full((self.window_height, self.window_width, 3),
                                    self.C_DARK, dtype=np.uint8)

                # Draw keyboard layout onto canvas
                self.keyboard.draw_keyboard(kb_canvas)

                right_hand_index = None
                left_hand_index  = None

                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        self.mp_draw.draw_landmarks(
                            camera_img, hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        label = results.multi_handedness[idx].classification[0].label
                        if label == self.HANDS_LABELS['Right']:
                            right_hand_index = idx
                        elif label == self.HANDS_LABELS['Left']:
                            left_hand_index  = idx
                logger.stage('landmarks')

                # ── mouse (right hand) ────────────────────────────────────────────
                if right_hand_index is not None:
                    self.mouse.handle_hand_gestures(results, right_hand_index, camera_img)
                else:
                    # Show "no right hand" status on camera
                    cv2.putText(camera_img, "RIGHT HAND: mouse",
                                (10, HEADER_H + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.C_ACCENT, 1, cv2.LINE_AA)
                    cv2.putText(camera_img, "not detected",
                                (10, HEADER_H + 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.C_GRAY, 1, cv2.LINE_AA)
                logger.stage('classify')

                # ── keyboard (left hand) ──────────────────────────────────────────
                if left_hand_index is not None:
                    self.keyboard.handle_hand_gestures(results, left_hand_index, kb_canvas)
                else:
                    cv2.putText(kb_canvas, "LEFT HAND: keyboard",
                                (10, HEADER_H + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.C_ACCENT, 1, cv2.LINE_AA)
                    cv2.putText(kb_canvas, "not detected",
                                (10, HEADER_H + 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.C_GRAY, 1, cv2.LINE_AA)
                logger.stage('gesture_map')

                # ── advance accuracy test when a new key is pressed ───────────────
                if self._test_running:
                    current_presses = self.keyboard.total_presses
                    if current_presses > prev_kb_presses:
                        prev_kb_presses = current_presses
                        self._advance_test()
                logger.stage('dispatch')

                # ── resize camera to match canvas height ──────────────────────────
                cam_h = self.window_height
                cam_w = int(cam_h * camera_img.shape[1] / camera_img.shape[0])
                camera_img = cv2.resize(camera_img, (cam_w, cam_h))

                # ── section labels ────────────────────────────────────────────────
                self._draw_section_label(camera_img, "◀  RIGHT HAND  |  MOUSE",
                                         10, HEADER_H + 2, self.C_ORANGE)
                self._draw_section_label(kb_canvas,  "LEFT HAND  |  KEYBOARD  ▶",
                                         10, HEADER_H + 2, self.C_GREEN)

                # ── combine panels ────────────────────────────────────────────────
                combined = np.zeros((self.window_height,
                                     cam_w + self.window_width, 3), dtype=np.uint8)
                combined[:, :cam_w]        = camera_img
                combined[:, cam_w:]        = kb_canvas

                # ── header + divider ──────────────────────────────────────────────
                self._draw_header(combined)
                self._draw_divider(combined, cam_w)

                # ── accuracy test overlay ─────────────────────────────────────────
                self._draw_test_overlay(combined)

                cv2.imshow(WINDOW_NAME, combined); logger.stage('render')

                key = cv2.waitKey(1)
                if key == ord('q') or cv2.getWindowProperty(
                        WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif key == ord('t'):
                    self._start_test()
                    prev_kb_presses = self.keyboard.total_presses
                elif key == ord('s'):
                    self.keyboard.stop_accuracy_tracking()
                    self._test_running = False
                elif key == ord('a'):
                    # Prompt in terminal for expected key
                    ek = input("[Accuracy] Enter expected key: ").strip()
                    if ek:
                        self.keyboard.set_expected_key(ek)
                        if not self.keyboard.tracking_accuracy:
                            self.keyboard.start_accuracy_tracking()
                        print(f"[Accuracy] Expected key set to '{ek}'")
        finally:
            logger.save()
            cap.release()
            cv2.destroyAllWindows()

        # Print final report if tracking was active
        if self.keyboard.tracking_accuracy or self.keyboard.total_presses > 0:
            self.keyboard.print_accuracy_report()

if __name__ == "__main__":
    MouseAndKeyboard().start()
