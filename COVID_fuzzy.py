import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from takagi_sugeno_kang_fuzzy_system.main import main


# Read the numeric dataset
numeric_data = pd.read_csv("COVID_numerics.csv")
print(numeric_data.shape)

# Read the image dataset
image_data = np.loadtxt("COVID_IMG.csv", delimiter=',')

# Reshape each row into a 21x21 matrix if needed
image_matrices = image_data.reshape(-1, 21, 21)
print(image_matrices.shape)  # Should be (number_of_samples, 21, 21)

# Access the first binary image/matrix
first_image = image_matrices[5]  # Shape is (21, 21)

# Flatten the matrix into a 1D array
flattened = first_image.flatten()  # Shape becomes (441,)

"""
# Plot the binary array as a heatmap
plt.figure(figsize=(6, 6))
plt.imshow(first_image, cmap='binary', origin='lower')
plt.title("image")
plt.xlabel("ecg (k)")
plt.ylabel("ecg (k-delay)")
plt.grid(False)
plt.show()
"""

########## TAKAGI-SUGENO-SANG FUZZY SYSTEM##########
main("COVID_numerics.csv", 3)

