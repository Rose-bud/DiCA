import torch
from torch import nn
from torch.nn import functional as F
from utils.config import args

class ImageNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: dimension of tags
        :param output_dim: dimensionality of the final representation
        """
        super(ImageNet, self).__init__()
        self.module_name = "image_model"

        # full-conv layers
        mid_num = 4096
        self.fc1 = nn.Linear(input_dim, mid_num)
        self.fc2 = nn.Linear(mid_num, mid_num)

        self.fc3 = nn.Linear(mid_num, output_dim)

        self.fc_new = nn.Linear(output_dim, args.num_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        norm = torch.norm(x, dim=1, keepdim=True)
        x = x / norm

        x_new = x
        return x_new,x
