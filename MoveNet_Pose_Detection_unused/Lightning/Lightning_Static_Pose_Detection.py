import tensorflow as tf 
import tensorflow_hub as hub
import numpy as np 
import cv2 

# Load the MoveNet model from TensorFlow Hub 
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

# Define the mapping of keypoints to body parts
keypoint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                  'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    
# Define the connections between keypoints to draw lines for visualization
connections = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
               (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

# Function to perform pose detection on a static image
def detect_pose_static(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert image to RGB (MoveNet expects RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image to the expected input size of MoveNet
    image_resized = tf.image.resize_with_pad(tf.expand_dims(image_rgb, axis=0), 192, 192) #192 for lightning
    # Convert the resized image tensor to a NumPy array with dtype uint8
    image_np = image_resized.numpy().astype(np.int32)
    # Perform inference
    outputs = movenet.signatures["serving_default"](tf.constant(image_np))
    # Extract the keypoints
    keypoints = outputs['output_0'].numpy()
    # Return the keypoints
    return keypoints

# Function to visualize keypoints on a static image
def visualize_pose_static(image_path, keypoints):
    # Read the image
    image = cv2.imread(image_path)
    # Convert keypoints to numpy array
    keypoints = np.array(keypoints)
    #print("Shape of keypoints array:", keypoints.shape)
    # Ensure keypoints array has the expected shape
    if keypoints.shape == (1, 1, 17, 3):
        # Extract keypoints from the array
        keypoints = keypoints[0, 0]
        # Loop through each keypoint
        for kp in keypoints:
            # Extract x and y coordinates of the keypoint
            x = int(kp[1] * image.shape[1])
            y = int(kp[0] * image.shape[0])
            # Draw a circle at the keypoint position
            cv2.circle(image, (x, y), 12, (255, 0, 0), -1)  # Increase thickness and change color to blue
        # Draw lines connecting keypoints
        for connection in connections:
            start_point = (int(keypoints[connection[0], 1] * image.shape[1]),
                           int(keypoints[connection[0], 0] * image.shape[0]))
            end_point = (int(keypoints[connection[1], 1] * image.shape[1]),
                         int(keypoints[connection[1], 0] * image.shape[0]))
            cv2.line(image, start_point, end_point, (0, 0, 255), 8)  # Increase thickness and change color to red
        
        # Convert image back to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Show the image with keypoints and lines using cv2.imshow
        cv2.imshow("Image Window",image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Unexpected shape of keypoints array:", keypoints.shape)
static_image_path = r"C:\Users\kchao\Downloads\dasquat.jpg"
#Perform pose detection on static image
static_keypoints = detect_pose_static(static_image_path)
visualize_pose_static(static_image_path, static_keypoints)


