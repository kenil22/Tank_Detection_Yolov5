import cv2
import torch
import torch.nn as nn
from pytorch_func import get_vector
import json

with open("using_pytorch_fea_extr_tank_only.json","r") as json_file:
    features = json.load(json_file)

def yolo_inferencing(frame):

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='model\\best_14.pt')
    results = model(frame)
    # Get the bounding box coordinates
    bbox_coordinates = results.pandas().xyxy[0]
    if bbox_coordinates.empty:
        print("DataFrame is empty")
        return frame
    else:
        if bbox_coordinates['confidence'][0] > 0.30 :
            xmin = int(bbox_coordinates.iloc[0:1]['xmin'])
            ymin = int(bbox_coordinates.iloc[0:1]['ymin'])
            xmax = int(bbox_coordinates.iloc[0:1]['xmax'])
            ymax = int(bbox_coordinates.iloc[0:1]['ymax'])

            cropped_img = frame[ ymin:ymax, xmin:xmax ]

            tensor_array1 = get_vector(cropped_img)

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
            if max(temp_list) > 0.71:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                name = temp_dict[max(temp_list)]

                org = (500, 170)
                fontFace = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (0,255,25)
                lineType = cv2.LINE_4

                img_text = cv2.putText(frame, name, org, fontFace, fontScale, color, lineType)


                return img_text
            else:
                return frame
        else:
            return frame