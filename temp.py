import cv2
import torch
import torch.nn as nn
from pytorch_func import get_vector
import json

with open("using_pytorch_fea_extr_face_only.json","r") as json_file:
    features = json.load(json_file)

def face_inferencing(frame):
    face_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if (len(faces)) > 0:
        for (x, y, w, h) in faces:
            cropped_img = frame[y:y+h, x:x+w]
        tensor_array1 = get_vector(frame)
        
        temp_dict = dict()
        temp_list = list()
    
        for k,b in features.items():

            tensor_array2 = torch.tensor(b)
            # Using PyTorch Cosine Similarity
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_sim = cos(tensor_array1.unsqueeze(0),
                          tensor_array2.unsqueeze(0))
            temp_dict[cos_sim.tolist()[0]] = k
            temp_list.append(cos_sim.tolist()[0])
        if temp_list[0] > 0.7:
            print("**"*10)
            name = "Maj Gen ARS Kahlon"
            print(name)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            org = (500, 100)
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0,255,25)
            lineType = cv2.LINE_4
            img_text = cv2.putText(frame, name, org, fontFace, fontScale, color, lineType)
            return img_text
        else:
            # print("its not more than 40")
            # print(len(frame))
            return frame
    else:
        return frame
