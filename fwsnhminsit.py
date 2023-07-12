import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from cv2 import (VideoCapture, namedWindow, imshow, waitKey, destroyWindow, imwrite)
#GEt the fashionmnist dataswt
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
#turn them inti dataloaders
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64)
#Define the model
class F_MINIST_CNN(nn.Module):
    def __init__(self,input,hidden,output):
        super().__init__()
        self.stack_layers=nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=1, out_channels=hidden*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden*2),
            nn.Dropout(0.25)
        )
        self.classifier=nn.Sequential(
            nn.Linear(hidden*2*7*7, hidden*2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden*2, output)
        )
    def forward(self, x):
        x=self.stack_layers(x)
        x=self.classifier(x)
        return x
#Define the model
model=F_MINIST_CNN(input=1, hidden=32, output=10)
#Define the loss function
loss_fn=nn.CrossEntropyLoss()
#Define the optimizer
optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
#Define the training loop
def train(dataloader, model, loss_fn, optimizer):
    size=len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        #Compute prediction and loss
        pred=model(X)
        loss=loss_fn(pred, y)
        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch%100==0:
            loss, current=batch, loss.item()
            print(f"loss: {loss } [{current}/{size}]")
#Define the test loop
def test(dataloader, model, loss_fn):
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    model.eval()
    test_loss, correct=0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred=model(X)
            test_loss+=loss_fn(pred, y).item()
            correct+=(pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss/=num_batches
    correct/=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#Train the model
epochs=10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
#save the model
torch.save(model.state_dict(), "model.pth")