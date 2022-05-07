import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class BaseNetwork:
    """The base network that will be used for testing
    It builds off of EfficientNet and simply adds a couple of convolutional layers, batch norms, and one fc layer
    """

    def __init__(
        self,
        action_size=6 # There are 0 - 5 ISUP grades
    ):
        self.basenet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
        self.basenet.eval().to(device)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, action_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = F.relu(self.basenet(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.fc(x.view(x.size(0),  -1)))
        return self.head(x)

