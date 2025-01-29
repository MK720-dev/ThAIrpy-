import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First joint
    b = np.array(b)  # Middle joint
    c = np.array(c)  # Last joint
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    
    # Process pose
    result = pose.process(rgb_frame)

    rgb_frame.flags.writeable = True
    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract landmark points & visibility
        landmarks = result.pose_landmarks.landmark

        def get_landmark(name):
            lm = landmarks[name]
            return [lm.x, lm.y], lm.visibility > 0.5  # Returns position & visibility status

        # Upper Body
        shoulder_l, vis_shoulder_l = get_landmark(mp_pose.PoseLandmark.LEFT_SHOULDER)
        shoulder_r, vis_shoulder_r = get_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        elbow_l, vis_elbow_l = get_landmark(mp_pose.PoseLandmark.LEFT_ELBOW)
        elbow_r, vis_elbow_r = get_landmark(mp_pose.PoseLandmark.RIGHT_ELBOW)
        wrist_l, vis_wrist_l = get_landmark(mp_pose.PoseLandmark.LEFT_WRIST)
        wrist_r, vis_wrist_r = get_landmark(mp_pose.PoseLandmark.RIGHT_WRIST)
        
        # Lower Body
        hip_l, vis_hip_l = get_landmark(mp_pose.PoseLandmark.LEFT_HIP)
        hip_r, vis_hip_r = get_landmark(mp_pose.PoseLandmark.RIGHT_HIP)
        knee_l, vis_knee_l = get_landmark(mp_pose.PoseLandmark.LEFT_KNEE)
        knee_r, vis_knee_r = get_landmark(mp_pose.PoseLandmark.RIGHT_KNEE)
        ankle_l, vis_ankle_l = get_landmark(mp_pose.PoseLandmark.LEFT_ANKLE)
        ankle_r, vis_ankle_r = get_landmark(mp_pose.PoseLandmark.RIGHT_ANKLE)
        heel_l, vis_heel_l = get_landmark(mp_pose.PoseLandmark.LEFT_HEEL)
        heel_r, vis_heel_r = get_landmark(mp_pose.PoseLandmark.RIGHT_HEEL)
        foot_l, vis_foot_l = get_landmark(mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
        foot_r, vis_foot_r = get_landmark(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)

        # Core/Trunk
        mid_hip = [(hip_l[0] + hip_r[0]) / 2, (hip_l[1] + hip_r[1]) / 2]
        mid_shoulder = [(shoulder_l[0] + shoulder_r[0]) / 2, (shoulder_l[1] + shoulder_r[1]) / 2]

        # Calculate angles if all required joints are visible
        detected_movements = []

        # **Shoulder Movements**
        if vis_shoulder_l and vis_elbow_l and vis_hip_l:
            shoulder_l_angle = calculate_angle(elbow_l, shoulder_l, hip_l)
            if shoulder_l_angle > 90:
                detected_movements.append("Left Shoulder Abduction")
            if shoulder_l_angle < 40:
                detected_movements.append("Left Shoulder Flexion")

        if vis_shoulder_r and vis_elbow_r and vis_hip_r:
            shoulder_r_angle = calculate_angle(elbow_r, shoulder_r, hip_r)
            if shoulder_r_angle > 90:
                detected_movements.append("Right Shoulder Abduction")
            if shoulder_r_angle < 40:
                detected_movements.append("Right Shoulder Flexion")

        # **Elbow Movements**
        if vis_shoulder_l and vis_elbow_l and vis_wrist_l:
            elbow_l_angle = calculate_angle(shoulder_l, elbow_l, wrist_l)
            if elbow_l_angle < 100:
                detected_movements.append("Left Elbow Flexion")

        if vis_shoulder_r and vis_elbow_r and vis_wrist_r:
            elbow_r_angle = calculate_angle(shoulder_r, elbow_r, wrist_r)
            if elbow_r_angle < 100:
                detected_movements.append("Right Elbow Flexion")

        # **Knee Movements**
        if vis_hip_l and vis_knee_l and vis_ankle_l:
            knee_l_angle = calculate_angle(hip_l, knee_l, ankle_l)
            if knee_l_angle < 120:
                detected_movements.append("Left Knee Flexion")

        if vis_hip_r and vis_knee_r and vis_ankle_r:
            knee_r_angle = calculate_angle(hip_r, knee_r, ankle_r)
            if knee_r_angle < 120:
                detected_movements.append("Right Knee Flexion")

        # **Hip Movements**
        if vis_shoulder_l and vis_hip_l and vis_knee_l:
            hip_l_angle = calculate_angle(shoulder_l, hip_l, knee_l)
            if hip_l_angle > 170:
                detected_movements.append("Left Hip Extension")
            if hip_l_angle < 90:
                detected_movements.append("Left Hip Flexion")

        if vis_shoulder_r and vis_hip_r and vis_knee_r:
            hip_r_angle = calculate_angle(shoulder_r, hip_r, knee_r)
            if hip_r_angle > 170:
                detected_movements.append("Right Hip Extension")
            if hip_r_angle < 90:
                detected_movements.append("Right Hip Flexion")

        # **Trunk/Core Movements**
        if vis_shoulder_l and vis_shoulder_r and vis_hip_l and vis_hip_r:
            trunk_flexion = calculate_angle(mid_shoulder, mid_hip, [mid_hip[0], mid_hip[1] + 1])
            if trunk_flexion < 170:
                detected_movements.append("Trunk Flexion")
            if trunk_flexion > 190:
                detected_movements.append("Trunk Extension")

        # **Wrist Movements**
        if vis_elbow_l and vis_wrist_l:
            wrist_l_angle = calculate_angle(elbow_l, wrist_l, [wrist_l[0], wrist_l[1] - 0.1])
            if wrist_l_angle < 80:
                detected_movements.append("Left Wrist Flexion")
            if wrist_l_angle > 100:
                detected_movements.append("Left Wrist Extension")

        if vis_elbow_r and vis_wrist_r:
            wrist_r_angle = calculate_angle(elbow_r, wrist_r, [wrist_r[0], wrist_r[1] - 0.1])
            if wrist_r_angle < 80:
                detected_movements.append("Right Wrist Flexion")
            if wrist_r_angle > 100:
                detected_movements.append("Right Wrist Extension")

        # Display detected movements
        for i, movement in enumerate(detected_movements):
            cv2.putText(frame, movement, (10, 60 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
