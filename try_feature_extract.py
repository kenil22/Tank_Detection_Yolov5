import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

features_dict = dict()

model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        # my_embedding.copy_(o.data)
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding


import os
for img_name in os.listdir("image_folder"):
    img_path = os.path.join("image_folder",img_name)
    pic_one_vector = get_vector(img_path)
    features_dict[img_name] = pic_one_vector.tolist()

import json
with open("using_pytorch_fea_extr_tank_only.json","w") as f:
    json.dump(features_dict,f)

# pic_one_vector = get_vector("sample_image1.jpg")
# pic_two_vector = get_vector("sample_image2.jpg")

# x1 = pic_one_vector.tolist()
# x2 = pic_two_vector.tolist()

# tensor_array1 = torch.tensor(x1)
# tensor_array2 = torch.tensor(x2)

# # Using PyTorch Cosine Similarity
# cos = nn.CosineSimilarity(dim=1, eps=1e-6)
# cos_sim = cos(tensor_array1.unsqueeze(0),
#               tensor_array2.unsqueeze(0))
# print('\nCosine similarity: {0}\n'.format(cos_sim))