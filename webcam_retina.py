from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet34(pretrained=True) 
model.fc = nn.Linear(model.fc.in_features, 18) 
model.load_state_dict(torch.load(r"C:\Users\USER\Documents\ADEK\Semester 3 (2024-2025 okt-april)\studieproject\res34_fair_align_multi_7_20190809.pt", map_location=device)) 
model = model.to(device) 
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.299, 0.224, 0.225])])


cam = cv2.VideoCapture(0)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
age_groups = ["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","70+"]
frame_count = 0
results = {}

while True:
    ret, frame = cam.read()
    if not ret:
        break

frame_count += 1

if frame_count % 20 == 0:

    small_rgb = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    small_rgb = cv2.cvtColor(small_rgb, cv2.COLOR_BGR2RGB)
    
    results = RetinaFace.detect_faces(rgb_frame)

if isinstance(results, dict):
    for key, face in results.items():
        x1, y1, x2, y2 = [c * 4 for c in face["facial_area"]]
        
        face_img = rgb[y1:y2, x1:x2]

if face_img.size > 0:
    face_tensor = transform(face_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(face_tensor).cpu().numpy().squeeze()
        
        gender_pred = np.argmax(outputs[7:9])
        age_pred = np.argmax(outputs[9:18])

        gender = "Male" if gender_pred == 0 else "Female"
        age = age_groups[age_pred]
        label = f"{gender}, {age}"

        cv2.rectangle(frame, (x1,x2), (y1,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


out.write(frame)
cv2.imshow('Camera',frame)
    
if cv2.waitKey(1) == ord('q'):
    break

cam.release()
out.release()
cv2.destroyAllWindows()






