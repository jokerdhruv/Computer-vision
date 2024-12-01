import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# Step 1: Load the image
image = cv2.imread('Lena.png')  # Replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Reshape the image to a 2D array of pixels
flat_image = np.reshape(image, [-1, 3])

# Step 3: Estimate the bandwidth (window size W) for MeanShift
bandwidth = estimate_bandwidth(flat_image, quantile=0.1, n_samples=500)

# Step 4: Perform Mean Shift Clustering
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
mean_shift.fit(flat_image)

# Step 5: Retrieve the labels and cluster centers
labels = mean_shift.labels_
cluster_centers = mean_shift.cluster_centers_

# Step 6: Reshape the labels to match the original image
segmented_image = np.reshape(labels, image.shape[:2])

# Step 7: Display the segmented image
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')
plt.show()
