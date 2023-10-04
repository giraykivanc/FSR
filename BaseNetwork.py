import torch.nn as nn



class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(128 * 4 * 4, 128)  # Adjust the input size based on the pooling layers and input image size
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 10)

    def print_shape(self,input):
        print(input.shape)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = x.view(-1, 128*4*4)
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x