import cv2
import mediapipe as mp

# 1. Initialize MediaPipe Pose module (Legacy API works perfectly on 3.11)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 2. Open your video file
cap = cv2.VideoCapture("squat.mp4")

print("Analyzing squat... Press 'q' to quit.")

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
        # We multiply the AI's decimal output by the video width/height to get pixels
        hip = (int(landmarks[24].x * w), int(landmarks[24].y * h))
        knee = (int(landmarks[26].x * w), int(landmarks[26].y * h))
        ankle = (int(landmarks[28].x * w), int(landmarks[28].y * h))

        # Draw our custom neon green tracking lines
        cv2.line(frame, hip, knee, (0, 255, 0), 4)
        cv2.line(frame, knee, ankle, (0, 255, 0), 4)

        # Draw glowing red joints
        cv2.circle(frame, hip, 8, (0, 0, 255), -1)
        cv2.circle(frame, knee, 8, (0, 0, 255), -1)
        cv2.circle(frame, ankle, 8, (0, 0, 255), -1)

    # 7. Show the custom UI window
    cv2.imshow("Squat Tracker Pro", frame)

    # 8. Wait 20 milliseconds per frame, quit if 'q' is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()