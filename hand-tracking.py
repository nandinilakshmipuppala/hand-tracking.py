import cv2
import mediapipe as mp
import pyautogui
import math
import time

# -----------------------------
# SETUP
# -----------------------------

pyautogui.PAUSE = 0.05

screen_width, screen_height = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# Variables
dragging = False
last_scroll_y = None
last_left_click = 0
last_right_click = 0

# -----------------------------
# FUNCTIONS
# -----------------------------

def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


# -----------------------------
# MAIN LOOP
# -----------------------------

while True:

    success, frame = cap.read()

    if not success:
        print("Camera not found!")
        break

    # Flip camera
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # -----------------------------
            # GET LANDMARK POSITIONS
            # -----------------------------

            index = hand_landmarks.landmark[8]
            middle = hand_landmarks.landmark[12]
            thumb = hand_landmarks.landmark[4]

            index_x = int(index.x * frame.shape[1])
            index_y = int(index.y * frame.shape[0])

            middle_x = int(middle.x * frame.shape[1])
            middle_y = int(middle.y * frame.shape[0])

            thumb_x = int(thumb.x * frame.shape[1])
            thumb_y = int(thumb.y * frame.shape[0])

            # -----------------------------
            # DISTANCES
            # -----------------------------

            thumb_index = distance(
                thumb_x, thumb_y,
                index_x, index_y
            )

            thumb_middle = distance(
                thumb_x, thumb_y,
                middle_x, middle_y
            )

            index_middle = distance(
                index_x, index_y,
                middle_x, middle_y
            )

            # -----------------------------
            # 1. CURSOR MOVEMENT
            # -----------------------------

            # Move cursor when index finger is away
            # from thumb and middle finger
            if thumb_index > 50 and index_middle > 40:

                mouse_x = int(
                    index_x / frame.shape[1] * screen_width
                )

                mouse_y = int(
                    index_y / frame.shape[0] * screen_height
                )

                pyautogui.moveTo(mouse_x, mouse_y)

            # -----------------------------
            # 2. LEFT CLICK + DRAG
            # -----------------------------

            if thumb_index < 35:

                if not dragging:
                    dragging = True
                    pyautogui.mouseDown()

                    cv2.putText(
                        frame,
                        "DRAGGING",
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

            else:

                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # -----------------------------
            # 3. RIGHT CLICK
            # -----------------------------

            if thumb_middle < 35 and thumb_index > 45:

                current_time = time.time()

                if current_time - last_right_click > 0.7:

                    pyautogui.rightClick()

                    last_right_click = current_time

                    cv2.putText(
                        frame,
                        "RIGHT CLICK",
                        (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2
                    )

            # -----------------------------
            # 4. LEFT CLICK
            # -----------------------------

            # Small pinch = left click
            # Only when middle finger is not close
            if (
                thumb_index < 30
                and thumb_middle > 45
            ):

                current_time = time.time()

                if current_time - last_left_click > 0.7:

                    # If not dragging, click
                    if not dragging:
                        pyautogui.click()

                    last_left_click = current_time

            # -----------------------------
            # 5. SCROLL
            # -----------------------------

            # Index + middle fingers close together
            if index_middle < 45 and thumb_index > 50:

                if last_scroll_y is not None:

                    movement = index_y - last_scroll_y

                    # Scroll UP
                    if movement < -8:
                        pyautogui.scroll(2)

                    # Scroll DOWN
                    elif movement > 8:
                        pyautogui.scroll(-2)

                last_scroll_y = index_y

            else:
                last_scroll_y = None

            # -----------------------------
            # SHOW DISTANCES
            # -----------------------------

            cv2.putText(
                frame,
                f"Index-Thumb: {int(thumb_index)}",
                (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Index-Middle: {int(index_middle)}",
                (20, 480),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    # -----------------------------
    # DISPLAY
    # -----------------------------

    cv2.putText(
        frame,
        "Virtual Mouse",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Q = Exit",
        (20, 520),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI Virtual Mouse", frame)

    # -----------------------------
    # EXIT
    # -----------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# -----------------------------
# CLEANUP
# -----------------------------

if dragging:
    pyautogui.mouseUp()

cap.release()
cv2.destroyAllWindows()
hands.close()

print("Virtual Mouse stopped successfully!")