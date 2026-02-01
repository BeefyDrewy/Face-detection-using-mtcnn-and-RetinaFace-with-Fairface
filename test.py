# from mtcnn import MTCNN
# from mtcnn.utils.images import load_image

# # Create a detector instance
# detector = MTCNN(device="CPU:0")

# # Load an image
# image = load_image("ivan.jpg")

# # Detect faces in the image
# result = detector.detect_faces(image)

# # Display the result
# print(result)

# ==================================================================================

from mtcnn import MTCNN
from mtcnn.utils.images import load_image
from mtcnn.utils.plotting import plot
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the image
image = load_image("ivan.jpg")

# Initialize MTCNN detector
mtcnn = MTCNN(device="CPU:0")

# Detect faces and landmarks
result = mtcnn.detect_faces(image, threshold_onet=0.7)
# Visualize the results
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis("off")

ax = plt.gca()
for face in result:
    x, y, w, h = face["box"]
    rect = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=3,
        edgecolor="red",
        facecolor="none"
    )
    ax.add_patch(rect)
    for key, (x, y) in face["keypoints"].items():
        plt.scatter(x, y, s=40, c="yellow")
    
# Plot the results
plt.imshow(plot(image, result))
print(result)
print("Faces detected:", len(result))
plt.show()