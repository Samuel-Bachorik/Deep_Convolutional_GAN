import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from graphing_class import CreateGraph
from models import Generator, Discriminator
from dataset_loader import ImagesLoader

start_time = time.time()
plt.rcParams['image.cmap'] = 'gray'

training_paths = []

#Append all training folders
for i in range(10):
    training_paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - training/{}/".format(i))

if __name__ == '__main__':

    #Method for showing images from model
    def show_images(images):
        sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

        for index, image in enumerate(images):
            plt.subplot(sqrtn, sqrtn, index+1)
            plt.imshow(image.reshape(28, 28))

    # Discriminator and generator loss, we use BCEWithLogitsLoss instead of BCELoss for better stability
    # Do not use sigmoid activation at the end of discriminator
    d_loss_function = nn.BCEWithLogitsLoss()
    g_loss_function = nn.BCEWithLogitsLoss()

    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)

    # Set training parameters
    epochs = 200
    lr = 0.0002
    batch_size = 512
    channels_noise = 100

    # Create models
    G = Generator(channels_noise= channels_noise, channels_img=1).to(device)
    D = Discriminator(input_shape= (1, 28, 28)).to(device)

    # Load dataset
    loader = ImagesLoader(batch_size)
    dataset = loader.get_dataset(training_paths, training=True)

    # Create optimizers for models
    g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Create graphing class
    Loss_chart = CreateGraph(118, "Generator and discriminator loss")

    # Train
    for epoch in range(epochs):

        #Batch loop
        for images, labels in (zip(*dataset)):

            real_inputs = images.to(device)

            real_outputs = D(real_inputs)
            real_label = torch.ones(real_inputs.shape[0], 1).to(device)

            noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
            fake_inputs = G(noise)

            fake_outputs = D(fake_inputs)
            fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

            outputs = torch.cat((real_outputs, fake_outputs), 0)
            targets = torch.cat((real_label, fake_label), 0)

            # Backward propagation
            d_loss = d_loss_function(outputs, targets)
            Loss_chart.num_for_D += float(d_loss.item())

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Generator
            noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
            fake_inputs = G(noise)
            fake_outputs = D(fake_inputs)

            # Backward propagation
            targets = torch.ones([fake_outputs.shape[0], 1])
            targets = targets.to(device)
            g_loss = g_loss_function(fake_outputs, targets)
            Loss_chart.num_for_G += float(g_loss.item())

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        print('[{}/{}] Average G loss: {:.3f}  Average D loss: {:.3f} '.format(epoch + 1, epochs, Loss_chart.num_for_G / 118,  Loss_chart.num_for_D / 118))

        # Show images
        imgs_numpy = ((fake_inputs.view(-1, 784)).data.cpu().numpy()+1.0)/2.0
        show_images(imgs_numpy[:16])
        plt.show()

        # Create graph of loss
        Loss_chart.count(epoch + 1)

        # Save model every 50th epoch
        if epoch % 50 == 0:
            torch.save(G, '0.0002_-1-1_Generator_epoch_{}.pth'.format(epoch + 1))
            print('Model saved at {} epoch'.format(epoch + 1))


    print('Training finished sucessfully.')
    print('Training took: {} seconds'.format(time.time()-start_time))
