import cv2
import numpy as np
import networkx as nx
from skimage import img_as_float
from skimage.color import rgb2gray
from networkx.algorithms.flow import minimum_cut

# Step 1: Load the image and initialize
image = cv2.imread('Rename.jpg')
gray_image = img_as_float(rgb2gray(image))  # Convert to grayscale for simplicity
h, w = gray_image.shape

# Initialize a graph
G = nx.Graph()

# Step 2: Initialize neighborhood pixels and calculate dissimilarity factors
def pixel_id(x, y):
    return x * w + y

# Add nodes and edges for pixel neighborhood
for i in range(h):
    for j in range(w):
        if i < h - 1:
            diff_v = abs(gray_image[i, j] - gray_image[i + 1, j])
            G.add_edge(pixel_id(i, j), pixel_id(i + 1, j), weight=diff_v)
        if j < w - 1:
            diff_h = abs(gray_image[i, j] - gray_image[i, j + 1])
            G.add_edge(pixel_id(i, j), pixel_id(i, j + 1), weight=diff_h)

# Step 3: Calculate sigma (standard deviation of pixel intensities)
sigma = np.std(gray_image)

# Step 4: Calculate weights (edge weights normalized by sigma)
for u, v, data in G.edges(data=True):
    data['weight'] = np.exp(-data['weight'] ** 2 / (2 * sigma ** 2))

# Step 5: Perform min-cut operation (using an arbitrary source and sink node)
source = pixel_id(0, 0)  # Assume the top-left corner as source
sink = pixel_id(h - 1, w - 1)  # Assume the bottom-right corner as sink

cut_value, partition = minimum_cut(G, source, sink, capacity='weight')

# Get the nodes from one partition (foreground)
reachable_nodes, non_reachable_nodes = partition

# Create segmented output
segmented_image = np.zeros((h, w))
for node in reachable_nodes:
    x = node // w
    y = node % w
    segmented_image[x, y] = gray_image[x, y]

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
