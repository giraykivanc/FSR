import torch
import torch.nn as nn

class Block2(nn.Module):
    def __init__(self,model,freeze):
        super(Block2, self).__init__()
        self.model = model
        self.freeze = freeze
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(32, 128, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.fc3 = nn.Linear(128 * 4 * 4, 128)  # Adjust the input size based on the pooling layers and input image size
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 10)

    def print_shape(self,input):
        print(input.shape)


    def reduceV2(self,tensor):
        reshaped_tensor = tensor.view(tensor.shape[0],(int)(tensor.shape[1] / 2), 2, tensor.shape[2], tensor.shape[3])
        # Merge the 2 chunks along the 0th dimension by taking the mean
        reduced_tensor = reshaped_tensor.mean(dim=2)
        return reduced_tensor


    
    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                x = self.model.pool1(self.model.relu1(self.model.conv1(x)))
                x = self.model.pool2(self.model.relu2(self.model.conv2(x)))
        else:
            x = self.model.pool1(self.model.relu1(self.model.conv1(x)))
            x = self.model.pool2(self.model.relu2(self.model.conv2(x)))
        x = self.reduceV2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = x.view(-1, 128*4*4)
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x