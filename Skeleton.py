import cv2
import math
import serial
import mediapipe as mp
from ultralytics import YOLO

# --- ARDUINO CONNECTION ---
# Change 'COM3' to your Arduino's port (e.g., '/dev/ttyUSB0' or 'COM4')
try:
    arduino = serial.Serial('/dev/cu.usbserial-1110', 115200, timeout=0.01)
    print("Arduino connected successfully!")
except Exception as e:
    print(f"Arduino not found: {e}")
    arduino = None


# --- HELPER FUNCTIONS ---

def count_left_fingers(landmarks):
    """Returns the total number of fingers held up on the left hand"""
    count = 0

    # Left Thumb (x-coordinate check)
    if landmarks.landmark[4].x > landmarks.landmark[3].x:
        count += 1

    # Other fingers (y-coordinate check: Tip vs PIP joint)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
            count += 1

    return count


def draw_pinch_beam_and_get_dist(frame, landmarks, width, height, color=(0, 255, 255)):
    thumb = landmarks.landmark[4]
    index = landmarks.landmark[8]

    x1, y1 = int(thumb.x * width), int(thumb.y * height)
    x2, y2 = int(index.x * width), int(index.y * height)

    cv2.line(frame, (x1, y1), (x2, y2), color, 3)
    cv2.circle(frame, (x1, y1), 5, color, cv2.FILLED)
    cv2.circle(frame, (x2, y2), 5, color, cv2.FILLED)

    return int(math.hypot(x2 - x1, y2 - y1))


# --- INITIALIZATION ---
model = YOLO('yolov8n.pt')
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(1)  # Using your secondary camera index

print("Starting camera... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    h, w, _ = frame.shape

    # Default states to send to Arduino
    right_speed = 0
    buzzer_state = 0

    # YOLO Detection
    results = model(frame, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 105, 180), 2)

    # MediaPipe Tracking
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    holistic_results = holistic.process(rgb_frame)

    # LEFT HAND (Controls Buzzer)
    if holistic_results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        left_finger_count = count_left_fingers(holistic_results.left_hand_landmarks)

        # Trigger buzzer if all 5 fingers are up
        if left_finger_count == 5:
            buzzer_state = 1

    # RIGHT HAND (Controls Stepper Motor Speed)
    if holistic_results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        right_dist = draw_pinch_beam_and_get_dist(frame, holistic_results.right_hand_landmarks, w, h, (0, 255, 255))

        if right_dist < 25:
            right_speed = 0
        else:
            mapped_speed = int((right_dist - 25) * 4.5)
            right_speed = min(mapped_speed, 1000)

            # SEND DATA TO ARDUINO
    if arduino:
        # Format: "Speed,BuzzerState\n"
        data_string = f"{right_speed},{buzzer_state}\n"
        arduino.write(data_string.encode())

    # HUD
    cv2.putText(frame, f'Motor Speed: {right_speed}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    buzzer_text = "ON" if buzzer_state == 1 else "OFF"
    cv2.putText(frame, f'Buzzer: {buzzer_text}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('AI Hardware Controller (Simplified)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
if arduino:
    arduino.write(b"0,0\n")  # Turn motor and buzzer off before closing
    arduino.close()
cap.release()
cv2.destroyAllWindows()
holistic.close()