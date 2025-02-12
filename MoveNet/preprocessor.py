import os
import shutil 
import csv 
import random 
import numpy as np
from movenet import process_image

# Funcrion to split a Poses folder into training and testing datasets
def split_dataset(input_folder, output_folder, train_ratio=0.6, seed=42):
  """
    Splits a dataset folder into training and testing sets.

    Args:
        input_folder (str): Path to the original dataset folder (e.g., '/Poses').
        output_folder (str): Path to the output folder (e.g., '/Output').
        split_ratio (float): Percentage of data used for training (default: 80%).
        seed (int): Random seed for reproducibility (default: 42).
    """
  random.seed(seed)

  # Ensure the output directory exists
  os.makedirs(output_folder, exist_ok=True)

  # Iterate through each pose category
  for pose_category in os.listdir(input_folder):
    pose_category_path = os.path.join(input_folder, pose_category)

    if not os.path.isdir(pose_category_path): # Skip files if any
      continue

    # Get all instances of a the pose category
    instances = sorted(os.listdir(pose_category_path)) # Sorting ensures consistency
    random.shuffle(instances) # Shuffle to randomize selection

    # Determine the split index
    split_index = int(len(instances) * train_ratio)

    # Split into train and test folders for the pose category
    training_instances = instances[:split_index]
    testing_instances = instances[split_index:]

    # Define output paths
    new_pose_category_path = os.path.join(output_folder, pose_category)
    os.makedirs(new_pose_category_path, exist_ok=True)
    train_folder = os.path.join(new_pose_category_path, f"training")
    test_folder = os.path.join(new_pose_category_path, f"testing")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Copy training instances
    for instance in training_instances:
      src_path = os.path.join(pose_category_path, instance)
      dst_path = os.path.join(train_folder, instance)
      if os.path.isfile(src_path): # Check if it's a file before copying
        shutil.copy(src_path, dst_path) # Copy instance

    # Copy testing instances
    for instance in testing_instances:
      src_path = os.path.join(pose_category_path, instance)
      dst_path = os.path.join(test_folder, instance)
      if os.path.isfile(src_path): # Check if it's a file before copying
        shutil.copy(src_path, dst_path) # Copy instance

    print(f"Processed {pose_category}: {len(training_instances)} training, {len(testing_instances)} testing.")


# Processes the dataset and writes keypoint data to training and testing CSVs
def process_dataset(data_path, output_train_csv_path, output_test_csv_path):

  # Define CSV headers
  csv_headers = ["pose", "image_name"] + [f"{kp}_{coord}" for kp in [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"] for coord in ["x", "y", "score"]]
  
  # Defining the path to the output CSV files: training_data.csv, testing_data.csv
  output_train_csv = os.path.join(output_train_csv_path, "training_data.csv")
  output_test_csv = os.path.join(output_test_csv_path, "testing_data.csv")

  # Initialize CSVs with headers
  for csv_file in [output_train_csv, output_test_csv]:
    with open(csv_file, 'w', newline='') as csvfile:
      csv_writer = csv.writer(csvfile)
      csv_writer.writerow(csv_headers)

  # Initialize an empty list to store the processed data (pose name, image name, and keypoints)
  processed_data = []
  # Initialize an empty list to store pose names
  poses = []

  # Loop through dataset directory (Pose1, Pose2, Pose3,...)
  for pose_category in os.listdir(data_path):
    poses.append(pose_category)
    pose_path = os.path.join(data_path, pose_category)
    if not os.path.isdir(pose_path):
      continue # Skip files

    # Loop through both training and testing subfolders for each pose
    for split in ['training', 'testing']:
      split_path = os.path.join(pose_path, split)
      if not os.path.isdir(split_path):
        continue # Skip files

      # Choosing which CSV file to write to based on whether it's the pose's training or testing subset
      csv_file = output_train_csv if split == 'training' else output_test_csv
      with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Loop through each image in the split folder
        for image_name in os.listdir(split_path):
          # Make sure image file is of valid type
          if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(split_path, image_name)
            # Process the image and get keypoints
            keypoints_with_scores = process_image(image_path)

            print(f"Processing image: {image_name}")
            print(f"Keypoints with scores: {keypoints_with_scores}")

            # Handle cases where no keypoints are detected
            if keypoints_with_scores is None:
              print(f"Warning: No keypoints detected for {image_name}. Skipping...")
              continue
            # Ensure it's a NumPy array before flattening
            keypoints = np.array(keypoints_with_scores)

            # Write pose label, image name, and extracted keypoints
            writer.writerow([pose_category] + [image_name] + keypoints.flatten().tolist())

  print("Data extraction and CSV writing complete!")


# Function calls to run movenet on dataset and generate CSVs

# Define dataset paths 
# Original dataset
input_folder = r"C:\Users\kchao\OneDrive\Documents\Dossier_Malek\ThAIrpy-\MoveNet\Pose_example_dataset"
# Path to new dataset after splitting 
output_folder = r"C:\Users\kchao\OneDrive\Documents\Dossier_Malek\ThAIrpy-\MoveNet\Pose_example_dataset_split"
# Split dataset (training, testing)
split_dataset(input_folder, output_folder)

# Defining folder path for the generated output CSVs 
output_train_csv_path= r"C:\Users\kchao\OneDrive\Documents\Dossier_Malek\ThAIrpy-\MoveNet"
output_test_csv_path = r"C:\Users\kchao\OneDrive\Documents\Dossier_Malek\ThAIrpy-\MoveNet"

# Process the dataset
process_dataset(output_folder, output_train_csv_path, output_test_csv_path)











