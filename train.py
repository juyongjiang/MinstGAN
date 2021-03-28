import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from network import TinyNet, Discriminator, Generator
from dataload import get_training_dataloader

os.makedirs('images', exist_ok=True)
os.makedirs('checkpoint', exist_ok=True)
checkpoint_path = os.path.join('checkpoint', '{epoch}-{loss}.pth')

def train(opt, dataloader_m=None):
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = TinyNet()
    discriminator = Discriminator()

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # Configure data loader
    os.makedirs('./data/mnist', exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
                                            datasets.MNIST('./data/mnist', 
                                                           train=True, 
                                                           download=True,
                                                           transform=transforms.Compose([
                                                                                            transforms.ToTensor(),
                                                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                                                        ])
                                                            ),
                                            batch_size=opt.batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.9, 0.99))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.9, 0.99))

    # ----------
    #  Training
    # ----------
    for epoch in range(opt.n_epochs):
        for i, (imgs, imgs_ns, labels) in enumerate(dataloader):
            
            # Configure input
            real_imgs = Variable(labels.type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            # Sample noise as generator input
            #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))) # [64, 256]
            # Generate a batch of images
            gen_imgs = generator(imgs.cuda()) + imgs_ns.cuda() #[64, 64, 64]
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                                            d_loss.item(), g_loss.item()))

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)
                weights_path = checkpoint_path.format(epoch=batches_done, loss=g_loss)
                print('saving generator weights file to {}'.format(weights_path))
                torch.save(generator.state_dict(), weights_path)
               

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--path', type=str, default='./dataset', help='path for dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=256, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
    opt = parser.parse_args()
    print(opt)

    dataloader = get_training_dataloader(opt.path)

    train(opt, dataloader)