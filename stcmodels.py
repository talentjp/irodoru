import torch
import torch.nn as nn
import torch.nn.functional as F
      
class Discriminator(nn.Module):
    input_img_depth = 3
    hint_img_depth = 3
    kernel_size = 3
    padding = int(kernel_size / 2) #'same' padding
    def __init__(self, image_size=256, conv1_dim=32):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.conv1_dim = conv1_dim
        self.conv1 = nn.Conv2d(self.input_img_depth + self.hint_img_depth, self.conv1_dim, self.kernel_size, stride = 1, padding = self.padding)
        self.batchnorm1 = nn.BatchNorm2d(self.conv1_dim)
        self.conv2 = nn.Conv2d(self.conv1_dim, self.conv1_dim * 2, self.kernel_size, stride = 2, padding = self.padding) #128 x 128 x 128
        self.batchnorm2 = nn.BatchNorm2d(self.conv1_dim * 2)
        self.conv3 = nn.Conv2d(self.conv1_dim * 2, self.conv1_dim * 4, self.kernel_size, stride = 2, padding = self.padding) #256 x 64 x 64
        self.batchnorm3 = nn.BatchNorm2d(self.conv1_dim * 4)
        self.conv4 = nn.Conv2d(self.conv1_dim * 4, self.conv1_dim * 8, self.kernel_size, stride = 2, padding = self.padding) #512 x 32 x 32     
        self.batchnorm4 = nn.BatchNorm2d(self.conv1_dim * 8)
        self.conv5 = nn.Conv2d(self.conv1_dim * 8, self.conv1_dim * 2, self.kernel_size, stride = 2, padding = self.padding) #128 x 16 x 16
        self.batchnorm5 = nn.BatchNorm2d(self.conv1_dim * 2)
        self.conv6 = nn.Conv2d(self.conv1_dim * 2, int(self.conv1_dim / 2), self.kernel_size, stride = 2, padding = self.padding) #32 x 8 x 8                            
        self.fc1 = nn.Linear(int(self.conv1_dim / 2) * int(self.image_size / 32 * self.image_size / 32), 1)
        
    def forward(self, x):
        x = F.leaky_relu(self.batchnorm1(self.conv1(x)))
        x = F.leaky_relu(self.batchnorm2(self.conv2(x)))
        x = F.leaky_relu(self.batchnorm3(self.conv3(x)))
        x = F.leaky_relu(self.batchnorm4(self.conv4(x)))
        x = F.leaky_relu(self.batchnorm5(self.conv5(x)))
        x = F.leaky_relu(self.conv6(x))
        x = x.view(-1, self.num_flat_features(x))                
        x = F.sigmoid(self.fc1(x))
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features                                                