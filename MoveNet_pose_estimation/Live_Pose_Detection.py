import tensorflow as tf 
import tensorflow_hub as hub
import numpy as np 
import cv2 
#from google.colab.patches import cv2_show 
import imageio 

# Load the MoveNet model from TensorFlow Hub 
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

# Define the mapping of keypoints to body parts
keypoint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                  'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    
# Define the connections between keypoints to draw lines for visualization
connections = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
               (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def detect_pose_live():
    # Open Webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize for MoveNet input
        frame_resized = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 192, 192)
        frame_np = frame_resized.numpy().astype(np.int32)

        # Perform inference
        outputs = movenet.signatures["serving_default"](tf.constant(frame_np))
        keypoints = outputs['output_0'].numpy()

        # Draw keypoints on frame
        draw_keypoints_and_connections(frame, keypoints)

        # Display the frame
        cv2.imshow('Live Pose Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def draw_keypoints_and_connections(frame, keypoints):
    keypoints = keypoints[0, 0]  # Extract keypoints

    for kp_index, kp in enumerate(keypoints):
        x = int(kp[1] * frame.shape[1])
        y = int(kp[0] * frame.shape[0])
        confidence = kp[2]

        if confidence > 0.3:  # Draw only confident keypoints
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Blue keypoints
            cv2.putText(frame, keypoint_names[kp_index], (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    for connection in connections:
        pt1 = (int(keypoints[connection[0], 1] * frame.shape[1]),
               int(keypoints[connection[0], 0] * frame.shape[0]))
        pt2 = (int(keypoints[connection[1], 1] * frame.shape[1]),
               int(keypoints[connection[1], 0] * frame.shape[0]))

        if keypoints[connection[0], 2] > 0.3 and keypoints[connection[1], 2] > 0.3:
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Green connections
            

# Run the live pose detection
detect_pose_live()