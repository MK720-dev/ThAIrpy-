# Importing needed modules 
# for the dataset splitting into training and testing subsets 
import os 
import shutil 
import random

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
    train_folder = os.path.join(output_folder, f"{pose_category}_training")
    test_folder = os.path.join(output_folder, f"{pose_category}_testing")
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

#split_dataset(r"/content/Poses", r"/content/Poses_Split")

