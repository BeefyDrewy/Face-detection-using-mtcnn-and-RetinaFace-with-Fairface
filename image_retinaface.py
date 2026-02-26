from retinaface import RetinaFace
from retinaface.commons import logger
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

do_plotting = False

my_logger = logger.Logger(r"C:\Users\USER\retinaface\Retinaface_image.ipynb")

def int_tuple(t):
    return tuple(int(x) for x in t)

fitzpatrick_labels = {
    0: "Type I - Zeer licht",
    1: "Type II - Licht",
    2: "Type III - Lichtbruin",
    3: "Type IV - Olijfkleurig/Bruin",
    4: "Type V - Donkerbruin",
    5: "Type VI - Zeer donker/Zwart"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 18)
model_path = r"C:\Users\USER\Documents\ADEK\Semester 3 (2024-2025 okt-april)\studieproject\res34_fair_align_multi_7_20190809.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def predict_fitzpatrick(face_crop):

    face_tensor = transform(face_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(face_tensor)
        prediction = outputs.cpu().numpy().squeeze()

    return np.argmax(prediction)


img_path = r"C:\Users\USER\Documents\ADEK\Semester 3 (2024-2025 okt-april)\studieproject\images\img3.jpg"
img = cv2.imread(img_path) 

if img is None:
    print("Image not found :(")
    
result = RetinaFace.detect_faces(img_path, threshold=0.1)

if result is None:
    print("No faces detected at all.")
    result = {}
    
else:
    print(f"Success! Detected {len(result.keys())} faces.")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), # ResNet standaard formaat
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for idx, identity in result.items():
    facial_area = identity["facial_area"]

    face_crop = img[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]

    if face_crop.size > 0:

        avg_color_per_row = np.average(face_crop, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)

        brightness = np.mean(avg_color)

        if brightness > 200: fitz = "Type I"
        elif brightness > 150: fitz = "Type III"
        else: fitz = "Type V/VI"

    #    cv2.putText(img, f"Fitzpatrick: {fitz}", (facial_area[0], facial_area[1]-40),
                   # cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        skin_type_idx = predict_fitzpatrick(face_crop)
        skin_label = fitzpatrick_labels.get(skin_type_idx, "Unknown")
  #      print(f"Face {idx} Skin Type: {skin_label}")

#        label_y = facial_area[1] - 10 if facial_area[1] > 20 else facial_area[1]
#        cv2.putText(img, f"skin: {skin_label}", (facial_area[0], label_y),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


for idx, identity in result.items():
    confidence = identity["score"]
    print(f"Face {idx} confidence: {confidence}")

    rectangle_color = (255, 0, 255)

    landmarks = identity["landmarks"]
    diameter = 1

    cv2.circle(img, int_tuple(landmarks["left_eye"]), diameter, (189, 183, 107), -1)
    cv2.circle(img, int_tuple(landmarks["right_eye"]), diameter, (189, 183, 107), -1)
    cv2.circle(img, int_tuple(landmarks["nose"]), diameter, (0, 255, 0), -1)
    cv2.circle(img, int_tuple(landmarks["mouth_left"]), diameter, (0, 0, 225), -1)
    cv2.circle(img, int_tuple(landmarks["mouth_right"]), diameter, (0, 0, 225), -1)

    facial_area = identity["facial_area"]

    cv2.rectangle(
    img,
    (facial_area[2], facial_area[3]),

    (facial_area[0], facial_area[1]),
    rectangle_color,
    1,)


if do_plotting is True:
    plt.imshow(img[:, :, ::-1])
    plt.axis("off")
    plt.show()


height, width, channels = img.shape
bottom_panel_height = 120

canvas = np.zeros((height + bottom_panel_height, width, channels), dtype=np.uint8)

canvas[0:height, 0:width] = img

info_text_1 = f"Gedetecteerd: {len(result.keys())} gezicht(en)"
info_text_2 = f"Fitzpatrick: {skin_label}" # Deze komt uit je predict_fitzpatrick functie
info_text_3 = "Project: Studieproject Fitzpatrick Analyse"

cv2.putText(canvas, info_text_1, (20, height + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(canvas, info_text_2, (20, height + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(canvas, info_text_3, (width - 350, height + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# Verzamel de data in de loop 
all_results = [] # Lijst om info per gezicht op te slaan

for idx, identity in result.items():
    facial_area = identity["facial_area"]
    
    # Voorspelling 
    face_crop = img[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
    if face_crop.size > 0:
        skin_type_idx = predict_fitzpatrick(face_crop)
        skin_label = fitzpatrick_labels.get(skin_type_idx, "Unknown")
        all_results.append(skin_label)
    
    # Teken alleen het kader en de punten 
    cv2.rectangle(img, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (255, 0, 255), 1)
    

# Dashboard maken 
h, w, c = img.shape
bottom_panel_height = 180 
canvas = np.zeros((h + bottom_panel_height, w, c), dtype=np.uint8)
canvas[0:h, 0:w] = img

line1 = f"Totaal gedetecteerd: {len(result)} gezichten"

#Info in het zwarte balkje zetten
from collections import Counter
counts = Counter(all_results)

summary_list = [f"{label}: {count}x" for label, count in counts.items()]

line2 = " | ".join(summary_list[:4])
line3 = " | ".join(summary_list[4:])

y_start = h+ 40
for i, text in enumerate([line1, line2, line3]):
    cv2.putText(
        canvas,
        text,
        (20, y_start + (i * 45)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)



#cv2.putText(canvas, f"Totaal gedetecteerd: {len(result)} gezichten", (20, h + 35), 
          #  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Samenvatting van de Fitzpatrick types
#summary_text = "Huidtypes: "
#for label, count in counts.items():
#    if count > 0:
   #     summary_text += f"{label}: {count}x | "

# Samenvatting op de tweede regel (iets kleiner als het veel is)
#cv2.putText(canvas, summary_text[:100], (20, h + 80), 
   #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)



cv2.putText(canvas, "Fitzpatrick Huidtype Analyse", (w - 350, h + 130), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

display_img = canvas

cv2.imwrite("output_detected.jpg", display_img)
print("Saved as output_detected.jpg")

cv2.imshow("Detected Faces", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()




