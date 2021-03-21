import numpy as np
import os
from PIL import Image


def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    image = Image.open(image_path)
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor
#

# print(x.shape)

import torch

traindir = '/content/data/intel_data/train'


class inference_sqeeze:
    def __init__(self,model_weight, train_dir,class_name):
        self.model_weight = model_weight
        self.train_dir = train_dir
        self.categories = []
        self.class_name = class_name

        for d in os.listdir(self.train_dir):
            self.categories.append(d)
        print(self.categories)

    def inference_on_image(self,img_path):
        model = torch.load(self.model_weight,map_location='cuda')#torchvision.models.squeezenet1_0(pretrained = False, progress= True)
        x = process_image(img_path)
        pred_labels = model(torch.unsqueeze(x.cuda(), 0))
        with open(self.class_name, 'r') as file_handle:
            lines = file_handle.read().splitlines()
        #print(pred_labels)
        lines = np.array(pred_labels.cpu().detach().numpy()).ravel()
        index = list(lines).index(max(list(lines)))
        print("predicted labels is :", self.categories[index])
        return 0

    def inference_on_folder(self,folder_path):
        model = torch.load(self.model_weight, map_location='cuda')
        for filename in os.listdir(folder_path):
            #img = Image.open(os.path.join(folder_path, filename))
            x = process_image(os.path.join(folder_path,filename))
            pred_labels = model(torch.unsqueeze(x.cuda(), 0))
            with open(self.class_name, 'r') as file_handle:
              lines = file_handle.read().splitlines()
              lines = np.array(pred_labels.cpu().detach().numpy()).ravel()
              index = list(lines).index(max(list(lines)))
              print("predicted labels is :", self.categories[index])
        return 0






a = inference_sqeeze('/content/squeeze_net_model_1.pt', '/content/data/intel_data/train','/content/names.txt').inference_on_image('/content/data/intel_data/testdic/da.jpg')
b_folder = inference_sqeeze('/content/squeeze_net_model_1.pt', '/content/data/intel_data/train','/content/names.txt').inference_on_folder('/content/data/intel_data/testdic')