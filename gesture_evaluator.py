import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from virtual_mouse import VirtualMouse
from virtual_keyboard import VirtualKeyboard

class GestureEvaluator:
    # All gesture classes the system can detect
    GESTURE_CLASSES = ["neutral", "move", "left_click", "right_click", "click_hold"]

    def __init__(self):
        # mirror: whether to horizontally flip webcam frames for display
        self.mirror = True
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mouse = VirtualMouse(
            self.mp_hands, self.hands, self.mp_draw, 640, 480
        )
        self.keyboard = VirtualKeyboard(
            self.mp_hands, self.hands, self.mp_draw, 640, 480
        )

        self.y_true = []  # ground truth labels (entered by evaluator)
        self.y_pred = []  # model-predicted labels

    def detect_gesture(self, hand_landmarks, img_shape):
        """Returns the gesture label the model would produce."""
        landmarks, is_finger_up = self.mouse.get_finger_positions(
            hand_landmarks, img_shape
        )
        left_click, right_click, click_hold = self.mouse.detect_gestures(
            hand_landmarks, img_shape
        )

        if left_click:
            return "left_click"
        elif right_click:
            return "right_click"
        elif click_hold:
            return "click_hold"
        elif is_finger_up['index']:
            return "move"
        else:
            return "neutral"

    def run_evaluation(self):
        """
        Interactive evaluation session.
        The evaluator presses number keys to declare the TRUE gesture
        being performed, and the system simultaneously predicts it.

        Controls:
          1 = neutral   2 = move   3 = left_click
          4 = right_click   5 = click_hold   q = quit
        """
        cap = cv2.VideoCapture(0)
        label_map = {
            ord('1'): "neutral",
            ord('2'): "move",
            ord('3'): "left_click",
            ord('4'): "right_click",
            ord('5'): "click_hold"
        }

        print("Evaluation started.")
        print("Hold a gesture and press its number key to record a sample.")
        print("1=neutral  2=move  3=left_click  4=right_click  5=click_hold  q=quit")

        current_true = None
        SAMPLES_PER_LABEL = 50  # collect 50 samples per gesture for reliability

        while True:
            success, frame = cap.read()
            if not success:
                continue

            if self.mirror:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            predicted = "neutral"
            right_index = None
            left_index = None
            if results.multi_hand_landmarks:
                for i, hand_lm in enumerate(results.multi_hand_landmarks):
                    self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)
                    # Determine handedness label if available
                    hand_label = None
                    if results.multi_handedness and i < len(results.multi_handedness):
                        hand_label = results.multi_handedness[i].classification[0].label
                        # If display is mirrored, the visual left/right is swapped — correct for control mapping
                        if self.mirror:
                            hand_label = 'Left' if hand_label == 'Right' else 'Right'

                    # Map indices for keyboard (left) and mouse (right)
                    if hand_label == 'Right':
                        right_index = i
                    elif hand_label == 'Left':
                        left_index = i

                # Prefer using the right hand for gesture prediction (mouse gestures)
                if right_index is not None:
                    predicted = self.detect_gesture(results.multi_hand_landmarks[right_index], frame.shape)
                else:
                    # Fallback to first detected hand
                    predicted = self.detect_gesture(results.multi_hand_landmarks[0], frame.shape)

                # Detection-only: evaluate gestures without performing actions
                if right_index is not None:
                    right_lm = results.multi_hand_landmarks[right_index]
                    lclick, rclick, chold = self.mouse.detect_gestures(right_lm, frame.shape)
                    finger_positions, is_finger_up = self.mouse.get_finger_positions(right_lm, frame.shape)
                    status = "Mouse: "
                    if lclick:
                        status += "LeftClick"
                    elif rclick:
                        status += "RightClick"
                    elif chold:
                        status += "Hold"
                    elif is_finger_up.get('index', False):
                        status += "Move"
                    else:
                        status += "Neutral"
                    cv2.putText(frame, status, (10,120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

                if left_index is not None:
                    left_lm = results.multi_hand_landmarks[left_index]
                    # detect keyboard click (no action)
                    kclicked = self.keyboard.detect_click(left_lm)
                    # map fingertip to keyboard window coords and get hovered key
                    index_tip = left_lm.landmark[8]
                    fx = int(index_tip.x * self.keyboard.window_width)
                    fy = int(index_tip.y * self.keyboard.window_height)
                    hovered_key = self.keyboard.get_clicked_key((fx, fy))
                    kstatus = f"Keyboard: {'Clicking' if kclicked else 'Idle'}"
                    if hovered_key:
                        kstatus += f" {hovered_key}"
                    cv2.putText(frame, kstatus, (10,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 1)

            # Overlay instructions
            cv2.putText(frame, f"Predicted: {predicted}", (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples so far: {len(self.y_true)}", (10, 60),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
            cv2.putText(frame, "1=neutral 2=move 3=LC 4=RC 5=Hold Q=quit",
                        (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 200, 200), 1)

            cv2.imshow("Gesture Evaluator", frame)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            elif key in label_map:
                true_label = label_map[key]
                self.y_true.append(true_label)
                self.y_pred.append(predicted)
                print(f"Recorded: TRUE={true_label}  PRED={predicted}  "
                      f"({'CORRECT' if true_label == predicted else 'WRONG'})")

        cap.release()
        cv2.destroyAllWindows()

    def compute_metrics(self):
        """Compute and display all classification metrics."""
        if len(self.y_true) < 10:
            print("Not enough samples. Collect at least 10 before evaluating.")
            return

        classes = self.GESTURE_CLASSES

        print("\n" + "="*60)
        print("GESTURE RECOGNITION EVALUATION RESULTS")
        print("="*60)

        # Overall accuracy
        acc = accuracy_score(self.y_true, self.y_pred)
        print(f"\nOverall Accuracy:  {acc*100:.2f}%")
        print(f"Total Samples:     {len(self.y_true)}")

        # Per-class metrics
        print("\nPer-Gesture Breakdown:")
        print(classification_report(
            self.y_true, self.y_pred,
            labels=classes,
            target_names=classes,
            zero_division=0
        ))

        # Macro averages
        precision = precision_score(self.y_true, self.y_pred,
                                    average='macro', labels=classes,
                                    zero_division=0)
        recall    = recall_score(self.y_true, self.y_pred,
                                 average='macro', labels=classes,
                                 zero_division=0)
        f1        = f1_score(self.y_true, self.y_pred,
                             average='macro', labels=classes,
                             zero_division=0)

        print(f"Macro Precision:  {precision*100:.2f}%")
        print(f"Macro Recall:     {recall*100:.2f}%")
        print(f"Macro F1 Score:   {f1*100:.2f}%")

        # Confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred, labels=classes)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes
        )
        plt.title('Gesture Recognition Confusion Matrix')
        plt.ylabel('True Gesture')
        plt.xlabel('Predicted Gesture')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        plt.show()
        print("\nConfusion matrix saved to confusion_matrix.png")

        # Per-gesture error analysis
        print("\nError Analysis:")
        for i, true_class in enumerate(classes):
            row = cm[i]
            total = row.sum()
            if total == 0:
                continue
            correct = cm[i][i]
            wrong = total - correct
            if wrong > 0:
                confused_with = [
                    (classes[j], cm[i][j])
                    for j in range(len(classes))
                    if j != i and cm[i][j] > 0
                ]
                print(f"  {true_class}: {wrong}/{total} errors → "
                      f"confused with {confused_with}")

if __name__ == "__main__":
    evaluator = GestureEvaluator()
    evaluator.run_evaluation()
    evaluator.compute_metrics()