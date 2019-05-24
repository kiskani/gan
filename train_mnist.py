import torch
from models import *
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

batch_size = 512
iterations = 3
discriminator_update_steps_k = 1
learning_rate = 0.01
momentum = 0.01
log_interval = 10

mnist_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(mnist_trainset, batch_size=batch_size) 

generator = Generator()
discriminator = Discriminator()
gen_optimizer = optim.SGD(generator.parameters(), lr=learning_rate, momentum=momentum)
dis_optimizer = optim.SGD(discriminator.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []

for epoch in range(iterations):
    for k in range(discriminator_update_steps_k):
        for i_batch, sample_batched in enumerate(train_dataloader):
            discriminator.zero_grad()

            x_data_input = sample_batched[0].view(sample_batched[0].size(0), -1)
            z_input = torch.randn([batch_size, input_noise_dim])
            x_fake_input = generator(z_input)

            fake_output = discriminator(x_fake_input)
            data_output = discriminator(x_data_input)

            disc_loss = -data_output.mean(dim=0)[0] - fake_output.mean(dim=0)[1]
            disc_loss.backward()

            dis_optimizer.step()

            if i_batch % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tDiscriminator\tLoss: {:.6f}'.format(
                    epoch, i_batch * batch_size, len(train_dataloader) * batch_size,
                    100. * i_batch / len(train_dataloader), disc_loss.item()))
                train_losses.append(disc_loss.item())
                train_counter.append(
                    (i_batch * batch_size) + ((epoch-1)*len(train_dataloader) * batch_size))
                torch.save(generator.state_dict(), "generator.pth")
                torch.save(discriminator.state_dict(), "discriminator.pth")


    for batch_idx, _ in enumerate(train_dataloader):
            generator.zero_grad()
            z_input = torch.randn([batch_size, input_noise_dim])
            x_fake_input = generator(z_input)
            fake_output = discriminator(x_fake_input)

            gen_loss = fake_output.mean(dim=0)[1]
            gen_loss.backward()
            gen_optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tGenerator\tLoss: {:.6f}'.format(
                    epoch, batch_idx * batch_size, len(train_dataloader) * batch_size,
                    100. * batch_idx / len(train_dataloader), gen_loss.item()))
                train_losses.append(gen_loss.item())
                train_counter.append(
                    (i_batch * batch_size) + ((epoch-1)*len(train_dataloader) * batch_size))
                torch.save(generator.state_dict(), "generator.pth")
                torch.save(discriminator.state_dict(), "discriminator.pth")
