# from mtcnn import MTCNN
# from mtcnn.utils.images import load_image
# from mtcnn.utils.plotting import plot
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Load the image
# image = load_image("../MTCNN/Work.jpg")

# # Initialize MTCNN detector
# mtcnn = MTCNN(device="CPU:0")

# # Detect faces and landmarks
# result = mtcnn.detect_faces(image, threshold_onet=0.7)
# # Visualize the results
# plt.figure(figsize=(10, 6))
# plt.imshow(image)
# plt.axis("off")

# ax = plt.gca()
# for face in result:
#     x, y, w, h = face["box"]
#     rect = patches.Rectangle(
#         (x, y),
#         w,
#         h,
#         linewidth=3,
#         edgecolor="red",
#         facecolor="none"
#     )
#     ax.add_patch(rect)
#     for key, (x, y) in face["keypoints"].items():
#         plt.scatter(x, y, s=40, c="yellow")
    
# # Plot the results
# plt.imshow(plot(image, result))
# print(result)
# print("Faces detected:", len(result))
# plt.show()

# ==========================================================

# import cv2
# from mtcnn import MTCNN

# # Initialize detector
# detector = MTCNN(device="CPU:0")

# # Open laptop camera (0 = default webcam)
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     raise RuntimeError("Could not open webcam")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert BGR (OpenCV) → RGB (MTCNN expects RGB)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Detect faces
#     results = detector.detect_faces(rgb_frame, threshold_onet=0.7)
#     frame = cv2.resize(frame, (640, 480))

#     # Draw detections
#     for face in results:
#         x, y, w, h = face["box"]

#         # Bounding box
#         cv2.rectangle(
#             frame,
#             (x, y),
#             (x + w, y + h),
#             (0, 0, 255),
#             2
#         )

#         # Landmarks
#         for (px, py) in face["keypoints"].values():
#             cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

#     # Show frame
#     cv2.imshow("MTCNN Live Face Detection", frame)

#     # Press Q to quit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()

# =========================================================

import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import transforms
from mtcnn import MTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FairFace model
model = torchvision.models.resnet34(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 18)
model.load_state_dict(torch.load("../mtcnn/res34_fair_align_multi_7_20190809.pt", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

detector = MTCNN(device="CPU:0")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb, threshold_onet=0.7)

    for face in results:
        x, y, w, h = face["box"]

        # Crop face safely
        face_img = rgb[y:y+h, x:x+w]
        if face_img.size == 0:
            continue

        # Preprocess
        face_tensor = transform(face_img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(face_tensor).cpu().numpy().squeeze()

        race_pred = np.argmax(outputs[:7])
        gender_pred = np.argmax(outputs[7:9])
        age_pred = np.argmax(outputs[9:18])

        gender = "Male" if gender_pred == 0 else "Female"
        age_groups = ["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","70+"]
        age = age_groups[age_pred]

        label = f"{gender}, {age}"

        # Draw
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Live Face Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

