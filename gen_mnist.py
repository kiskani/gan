import torch
import argparse
from torchvision.utils import save_image
from models import *
from datetime import datetime

parser = argparse.ArgumentParser(description='GAN MNIST Example')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
model = Generator().to(device)

PATH = "generator.pth"
model.load_state_dict(torch.load(PATH))

if __name__ == "__main__":
    current_time_str = "{}{}{}{}{}".format(datetime.now().month,
                                         datetime.now().day,
                                         datetime.now().hour,
                                         datetime.now().minute,
                                         datetime.now().second)
   
    with torch.no_grad():
        random_input = torch.randn(64, 180).to(device)
        sample = model(random_input).cpu()
        print(sample.size())
        one_sample = sample[0,:].view(1, 1, 28, 28)
        print(sample.size())
        print(one_sample.size())
        save_image(sample.view(64, 1, 28, 28),
                   'batch_sample_' + current_time_str + '.png')
        save_image(one_sample,
                   'one_sample_' + current_time_str + '.png')
