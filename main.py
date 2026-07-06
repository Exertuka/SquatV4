import cv2
import mediapipe as mp
import math
import sys
import os


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


# --- Helper Function to Calculate Torso Lean ---
def calculate_torso_lean(hip, shoulder):
    """
    Calculates the angle of the torso (hip to shoulder) relative to the vertical axis.
    """
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    # Absolute angle in degrees from vertical
    angle_rad = math.atan2(abs(dx), abs(dy))
    return math.degrees(angle_rad)


# 1. Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose

# 2. Now that mp_pose is defined, you can safely create the pose object
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7
)
# 2. Determine input and output video paths
if len(sys.argv) > 1:
    input_video = sys.argv[1]
else:
    input_video = "squat120.mp4"

if not os.path.exists(input_video):
    print(f"Error: Input video file '{input_video}' not found.")
    sys.exit(1)

# Open your video file
cap = cv2.VideoCapture(input_video)

# Get properties for VideoWriter
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Determine output filename: same folder and filename + _squat_analysis.mov
dir_name = os.path.dirname(input_video)
base_name, _ = os.path.splitext(os.path.basename(input_video))
output_video = os.path.join(dir_name, f"{base_name}_squat_analysis.mov")

# Initialize VideoWriter to export as MOV using macOS-compatible avc1 (H.264) codec
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print(f"Analyzing squat from '{input_video}'...")
print(f"Saving output to '{output_video}'... Press 'q' to quit.")


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

# --- Initialize Squat State Variables ---
rep_count = 0
depth_reached = False

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

        # --- Calculate the Knee and Hip Angles ---
        angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(knee, hip, shoulder)

        # --- Rep Counting & Depth Verification State Machine ---
        is_deep_enough = hip[1] > knee[1] or angle < 90
        if angle > 160:
            if depth_reached:
                rep_count += 1
                depth_reached = False
        elif is_deep_enough:
            depth_reached = True

        # --- Visual Feedback & Styling ---
        # Draw connections and joints with dynamic colors (Green when deep enough, otherwise Blue/Red)
        line_color = (0, 255, 0) if is_deep_enough else (255, 0, 0)
        joint_color = (0, 255, 0) if is_deep_enough else (0, 0, 255)

        # Draw tracking lines
        cv2.line(frame, hip, knee, line_color, 4)
        cv2.line(frame, knee, ankle, line_color, 4)
        cv2.line(frame, hip, shoulder, line_color, 4)

        # Draw joints
        cv2.circle(frame, hip, 8, joint_color, -1)
        cv2.circle(frame, knee, 8, joint_color, -1)
        cv2.circle(frame, ankle, 8, joint_color, -1)
        cv2.circle(frame, shoulder, 8, joint_color, -1)

        # Draw the visual angle arcs
        drawAngle(frame, hip, knee, ankle, angle)
        drawAngle(frame, knee, hip, shoulder, hip_angle)

        # --- HUD (Heads-Up Display) Overlay ---
        hud_bg = frame.copy()
        # Draw a semi-transparent black rectangle at top-left
        cv2.rectangle(hud_bg, (15, 15), (380, 160), (20, 20, 20), -1)
        cv2.addWeighted(hud_bg, 0.7, frame, 0.3, 0, frame)

        # 1. Reps Counter
        cv2.putText(frame, f"REPS: {rep_count}", (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

        # 2. Depth Feedback
        depth_text = "DEPTH: GOOD" if (is_deep_enough or depth_reached) else "DEPTH: SHALLOW"
        depth_color = (0, 255, 0) if (is_deep_enough or depth_reached) else (0, 165, 255)  # Green or Orange
        cv2.putText(frame, depth_text, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, depth_color, 2, cv2.LINE_AA)

        # 3. Posture Lean Feedback
        torso_angle = calculate_torso_lean(hip, shoulder)
        if torso_angle > 40:
            posture_text = "WARN: KEEP CHEST UP!"
            posture_color = (0, 0, 255)  # Red warning
        else:
            posture_text = f"POSTURE: OK ({int(torso_angle)}deg)"
            posture_color = (0, 255, 0)  # Green okay
        cv2.putText(frame, posture_text, (30, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, posture_color, 2, cv2.LINE_AA)
    # 7. Show the custom UI window (Fixed the 'frazme' typo here!)
    cv2.imshow("Squat Tracker", frame)
    
    # Save the current frame to the output video file
    out.write(frame)

    # 8. Wait 20 milliseconds per frame, quit if 'q' is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()