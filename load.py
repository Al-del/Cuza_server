import os

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from cv2 import (VideoCapture, namedWindow, imshow, waitKey, destroyWindow, imwrite)
class F_MINIST_CNN(nn.Module):
    def __init__(self,input_size, output_size,hidden_layers):
        super().__init__()
        self.layer_1_CNN=nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=hidden_layers, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_layers),
            nn.Dropout(0.25)
        )
        self.layer_2_CNN=nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers, out_channels=hidden_layers*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_layers*2),
            nn.Dropout(0.25)
        )
        self.classifier=nn.Sequential(
            nn.Linear(hidden_layers*2*7*7, hidden_layers*2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_layers*2, output_size)
        )
    def forward(self, x):
        x=self.layer_1_CNN(x)
        x=self.layer_2_CNN(x)
        x=x.view(x.size(0), -1)
        x=self.classifier(x)
        return x
#Load the model
model=F_MINIST_CNN(input_size=1, output_size=10, hidden_layers=32)
model.load_state_dict(torch.load("model.pth"))
model.eval()
class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
test_data=datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

def clothes():

    model=F_MINIST_CNN(input_size=1, output_size=10, hidden_layers=32)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    cam_port = 0
    cam = VideoCapture(cam_port)

    # reading the input using the camera
    result, image = cam.read()

    # If image will detected without any error,
    # show result
    if result:

        # showing result, it take frame name and image
        # output
        imshow("Haina", image)

        # saving image in local storage
        imwrite("test.jpg", image)

        # If keyboard interrupt occurs, destroy image
        # window
        destroyWindow("Haina")

    # If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")
    print("Image saved")
    img=Image.open("test.jpg")
    print("well idk")
    img=img.convert("L")
    img=img.resize((28,28))
    img=np.array(img)
    plt.imshow(img, cmap="gray")
    plt.show()
    #turn the photo to tensor
    img_tensor=ToTensor()(img)
    img_tensor=img_tensor.unsqueeze(0)
    #predict the class
    y_pred=model(img_tensor)
    print(y_pred.argmax(dim=1))
    print(class_names[y_pred.argmax(dim=1)])
    os.remove("test.jpg")
clothes()