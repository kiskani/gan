import torch
from models import *
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

batch_size = 512
iterations = 15
discriminator_update_steps_k = 1
learning_rate = 0.01
momentum = 0.01
log_interval = 10

gpu_id = torch.cuda.current_device()
cuda_name = torch.device(gpu_id)

mnist_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(mnist_trainset, batch_size=batch_size) 

generator = Generator()
discriminator = Discriminator()
gen_optimizer = optim.SGD(generator.parameters(), lr=learning_rate, momentum=momentum)
dis_optimizer = optim.SGD(discriminator.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []

criterion = torch.nn.BCELoss()

generator.cuda(cuda_name)
discriminator.cuda(cuda_name)

nn.init.normal_(generator.gen_fc1.weight, 0.0, 0.1)
nn.init.normal_(generator.gen_fc2.weight, 0.0, 0.1)
nn.init.normal_(generator.gen_fc3.weight, 0.0, 0.1)
nn.init.normal_(generator.gen_fc4.weight, 0.0, 0.1)
nn.init.normal_(generator.gen_fc5.weight, 0.0, 0.1)

nn.init.normal_(discriminator.dis_fc1.weight, 0.0, 0.1)
nn.init.normal_(discriminator.dis_fc2.weight, 0.0, 0.1)
nn.init.normal_(discriminator.dis_fc3.weight, 0.0, 0.1)

generator.train()
discriminator.train()

for epoch in range(iterations):
    for k in range(discriminator_update_steps_k):
        for i_batch, sample_batched in enumerate(train_dataloader):
            discriminator.zero_grad()

            x_data_input = sample_batched[0].view(sample_batched[0].size(0), -1).cuda(cuda_name)
            real_data_probs = discriminator(x_data_input)
            data_targets = torch.ones_like(real_data_probs)
            disc_loss_real = criterion(real_data_probs, data_targets)
            disc_loss_real.backward()
 
            z_input = torch.randn([x_data_input.size(0), input_noise_dim]).cuda(cuda_name)
            x_fake_input = generator(z_input).detach()    # detach to avoid training generator on these labels
            fake_probs = discriminator(x_fake_input)
            fake_targets = torch.zeros_like(fake_probs)
            disc_loss_fake = criterion(fake_probs, fake_targets)
            disc_loss_fake.backward()

            dis_optimizer.step()
            disc_loss = disc_loss_real + disc_loss_fake
 
            if i_batch % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tDiscriminator\tLoss: {:.6f}\tReal precision: {:.4f}\tFake precision: {:.4f}'.format(epoch, i_batch * batch_size, len(train_dataloader) * batch_size, 100. * i_batch / len(train_dataloader), disc_loss.item(),\
                      torch.mean(real_data_probs).item(), torch.mean(fake_probs).item()))
                train_losses.append(disc_loss.item())
                train_counter.append(
                    (i_batch * batch_size) + ((epoch-1)*len(train_dataloader) * batch_size))
                torch.save(generator.state_dict(), "generator.pth")
                torch.save(discriminator.state_dict(), "discriminator.pth")
                #print("Average batch probability of correct data discrimination:\tReal:{:.4f}\tFake:{:.4f}"\
                #       .format(torch.mean(real_data_probs).item(), torch.mean(fake_probs).item()))
 

    for batch_idx, _ in enumerate(train_dataloader):
            generator.zero_grad()
            z_input = torch.randn([batch_size, input_noise_dim]).cuda(cuda_name)
            x_fake_input = generator(z_input)
            fake_output = discriminator(x_fake_input)

            fake_targets = torch.ones_like(fake_output) 
            gen_loss = criterion(fake_output, fake_targets)
            gen_loss.backward()
            gen_optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tGenerator\tLoss: {:.6f}\tFake precision: {:.4f}'.format(
                    epoch, batch_idx * batch_size, len(train_dataloader) * batch_size,
                    100. * batch_idx / len(train_dataloader), gen_loss.item(), torch.mean(fake_output).item()))
                train_losses.append(gen_loss.item())
                train_counter.append(
                    (i_batch * batch_size) + ((epoch-1)*len(train_dataloader) * batch_size))
                torch.save(generator.state_dict(), "generator.pth")
                torch.save(discriminator.state_dict(), "discriminator.pth")
