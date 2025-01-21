import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the numeric dataset 
# (gender, age, marital_status,vacination,respiration class,heart rate,systolic blood pressure,temperature)
numeric_data = pd.read_csv("COVID_numerics.csv")

# Read the image dataset
image_data = np.loadtxt("COVID_IMG.csv", delimiter=',')


# Reshape each row into a 21x21 matrix if needed
image_matrices = image_data.reshape(-1, 21, 21)
print(image_matrices.shape)  # Should be (number_of_samples, 21, 21)