import torch.nn as nn
import torch.nn.functional as F

input_noise_dim = 100
input_feature_size = 784

generator_l2_units = 128
generator_l3_units = 256
generator_l4_units = 512
generator_l5_units = 1024

discriminator_l2_units = 512
discriminator_l3_units = 256
discriminator_l4_units = 1

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen_fc1 = nn.Linear(input_noise_dim, generator_l2_units)
        self.gen_fc2 = nn.Linear(generator_l2_units, generator_l3_units)
        self.gen_fc3 = nn.Linear(generator_l3_units, generator_l4_units)
        self.gen_fc4 = nn.Linear(generator_l4_units, generator_l5_units)
        self.gen_fc5 = nn.Linear(generator_l5_units, input_feature_size)

    def forward(self, x):
        x = F.relu(self.gen_fc1(x))
        x = F.relu(self.gen_fc2(x))
        x = F.relu(self.gen_fc3(x))
        x = F.relu(self.gen_fc4(x))
        x = F.relu(self.gen_fc5(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dis_fc1 = nn.Linear(input_feature_size, discriminator_l2_units)
        self.dis_fc2 = nn.Linear(discriminator_l2_units, discriminator_l3_units)
        self.dis_fc3 = nn.Linear(discriminator_l3_units, discriminator_l4_units)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.dis_fc1(x))
        x = F.relu(self.dis_fc2(x))
        x = F.relu(self.dis_fc3(x))
        x = self.sigmoid(x)
        return x
