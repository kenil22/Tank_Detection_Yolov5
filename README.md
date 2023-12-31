# Real-Time Tank Detection
---
Let's start with a thought that we have to build an application which will get a live feed from camera to **detect real-time tank**. In order to do that, I've used **state-of-the-art model YOLOV5** and perform **Transfer Learning** on it to solve my problem in short time.

**Transfer Learning on pre-trained model**
1. Collew Raw Data
2. Annotate data
   [Roboflow Website link](https://roboflow.com/)   
3. After annotating data, we can copy the raw url of dataset and use that link in jupyter notebook file to use the annotated dataset.  
4. Upload **YOLOV5.ipynb** to google colab and perform training and testing steps.
5. Once we are done with **transfer learning**, we can download the weights which can easily detect those tanks which we had given while re-training.
   
# Feature Extraction 
For feature extraction, I have used PyTorch framework and ResNet18 architecture to preprocess framework and extract meaning full features from given image. Refer to **pytorch_func.py** for detailed understanding.
   
# Building flow for inferencing.
1. Refer to **fina.py** which contains all flask routes. Look out for **generate_frames()** function which takes live feed(frame) as input and outputs frame with box in the frame where tank is located.
2. See file **using_pytorch_fea_extr_tank_only.json** in which I have stored extracted features of all tank images.

# Attaching some outputs

<img src="https://github.com/kenil22/Tank_Detection_Yolov5/assets/73990461/5d53723d-2229-4807-8360-76f92e4528ac" width="200" alt="Leopard2_Tank_Output">  

<img src="https://github.com/kenil22/Tank_Detection_Yolov5/assets/73990461/08eaa47d-ce73-4322-885f-222821d1906b" width="200" alt="AL_KHALID_OUTPUT">

<img src="https://github.com/kenil22/Tank_Detection_Yolov5/assets/73990461/579b4d29-df53-499e-a38f-28ad89d43de8" width="200" alt="SHAHPAR_UAV_OUTPUT">

<img src="https://github.com/kenil22/Tank_Detection_Yolov5/assets/73990461/7b2f75b5-3283-427f-ab59-248f0b648443" width="200" alt="T72_TANK_OUTPUT">


## To run this code :-

1. conda create -n <env-name> python==3.10.10
2. conda activate <env-name>
3. pip install -r requirements.txt 
   (Note : If this command stops running becasue of dependency issues in library. In that case, you can manually install that library using pip.)
4. python fina.py  
   

# Thank you for your time
