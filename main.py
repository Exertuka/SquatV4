import cv2
import mediapipe as mp
import math


# --- Helper Function to Calculate Angles ---
def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.
    a = hip, b = knee (vertex), c = ankle
    """
    # Calculate the angle in radians
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])

    # Convert radians to degrees
    angle = abs(radians * 180.0 / math.pi)

    # Ensure we get the interior angle (<= 180 degrees)
    if angle > 180.0:
        angle = 360 - angle

    return angle


# 1. Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose

# 2. Now that mp_pose is defined, you can safely create the pose object
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7
)
# 2. Open your video file
cap = cv2.VideoCapture("squat120.mp4")

print("Analyzing squat... Press 'q' to quit.")


def drawAngle(frame, a, b, c, angle):
    # Calculate absolute angles
    startAngle = math.degrees(math.atan2(a[1] - b[1], a[0] - b[0]))
    endAngle = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]))

    # Ensure startAngle and endAngle are within 0-360 for comparison
    startAngle = startAngle % 360
    endAngle = endAngle % 360

    # Calculate difference
    diff = endAngle - startAngle

    # If the arc is drawing the "long way" (>180), swap the angles
    if abs(diff) > 180:
        if diff > 0:
            diff -= 360
        else:
            diff += 360

    # Draw the arc based on the smallest difference
    cv2.ellipse(frame, b, (30, 30), 0, startAngle, startAngle + diff, (255, 255, 255), 2)

    # Draw the Angle Text
    text_position = (b[0] + 15, b[1])
    cv2.putText(frame, str(int(angle)), text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
while True:
    # 3. Read the frame from the video
    success, frame = cap.read()
    if not success:
        print("Video finished or not found.")
        break

    # 4. MediaPipe needs RGB colors, but OpenCV uses BGR. Convert it.
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 5. Process the image to find your joints
    results = pose.process(imgRGB)

    # 6. If it finds you, extract the coordinates and draw!
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get video dimensions to map AI coordinates to exact pixels on your screen
        h, w, _ = frame.shape

        # Extract Right Hip (24), Right Knee (26), and Right Ankle (28)
        hip = (int(landmarks[24].x * w), int(landmarks[24].y * h))
        knee = (int(landmarks[26].x * w), int(landmarks[26].y * h))
        ankle = (int(landmarks[28].x * w), int(landmarks[28].y * h))

        #extra points
        shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
        toes = (int(landmarks[31].x * w), int(landmarks[31].y * h))

        # --- Calculate the Knee Angle ---
        angle = calculate_angle(hip, knee, ankle)

        # Draw our custom neon green tracking lines
        cv2.line(frame, hip, knee, (255, 0, 0), 4)
        cv2.line(frame, knee, ankle, (255, 0, 0), 4)

        #extra lines
        cv2.line(frame, hip, shoulder, (255, 0, 0), 4)

        # Draw glowing red joints
        cv2.circle(frame, hip, 8, (0, 0, 255), -1)
        cv2.circle(frame, knee, 8, (0, 0, 255), -1)
        cv2.circle(frame, ankle, 8, (0, 0, 255), -1)

        #extra joints
        cv2.circle(frame, shoulder, 8, (0, 0, 255), -1)


        # Calculate the angle using your existing helper function
        angle = calculate_angle(hip, knee, ankle)

        # Call your new custom function
        drawAngle(frame, hip, knee, ankle, angle)

        #extra angle
        drawAngle(frame, knee, hip, shoulder, angle)
    # 7. Show the custom UI window (Fixed the 'frazme' typo here!)
    cv2.imshow("Squat Tracker", frame)

    # 8. Wait 20 milliseconds per frame, quit if 'q' is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()