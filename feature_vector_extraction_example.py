import pandas as pd
import numpy as np

# Load the Iris dataset from a URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Load CSV into DataFrame
df = pd.read_csv(url, header=None, names=column_names)

# Extract feature vectors (excluding the last column, which is the class)
feature_vectors = df.drop('class', axis=1).values  # Drop 'class' column and convert to NumPy array

# Ensure all values are of type float64
feature_vectors = feature_vectors.astype(np.float64)

# Print the shape and first feature vector
print("Feature vector shape:", feature_vectors.shape)
print("First feature vector:", feature_vectors[0])
