import torch
import torch.nn as nn


class CNNModel(nn.Module):
   def __init__(self):
       super(CNNModel, self).__init__()

       # CNN Feature Extractor
       self.features = nn.Sequential(
           nn.Conv2d(3, 64, kernel_size=3, padding=1),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2),

           nn.Conv2d(64, 128, kernel_size=3, padding=1),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2),

           nn.Conv2d(128, 256, kernel_size=3, padding=1),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2),
       )

       # Regression head
       self.regressor = nn.Sequential(
           nn.Flatten(),
           nn.Linear(256 * 32 * 32, 512), 
           nn.ReLU(),
           nn.Dropout(0.5),
           nn.Linear(512, 256),
           nn.ReLU(),
           nn.Dropout(0.5),
           nn.Linear(256, 2)  
       )

   def forward(self, x):
       x = self.features(x)
       coordinates = self.regressor(x)
       return coordinates