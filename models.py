import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,channels_noise, channels_img):
        super(Generator, self).__init__()
        #Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Model layers
        self.layers_generator = [

            self.transposed_conv_bn_act(channels_noise, 1024, (4, 4), (1, 1), (0, 0)),
            self.transposed_conv_bn_act(1024, 512, (4, 4), (1, 1), (1, 1)),
            self.transposed_conv_bn_act(512, 512, (4, 4), (1, 1), (1, 1)),
            self.transposed_conv_bn_act(512, 256, (4, 4), (1, 1), (1, 1)),
            self.transposed_conv_bn_act(256, 128, (4, 4), (2, 2), (1, 1))
        ]

        #Out layer with Tanh activation for better stability
        self.out_conv = nn.ConvTranspose2d(128, channels_img, kernel_size=(4,4), stride=(2,2), padding=(1,1))
        self.act = nn.Tanh()

        #Automatic weight init
        for i in range(len(self.layers_generator)):
            torch.nn.init.xavier_uniform_(self.layers_generator[i][0].weight)
            torch.nn.init.zeros_(self.layers_generator[i][0].bias)

        # Create network and put on device
        self.model_generator = nn.Sequential(*self.layers_generator)
        self.model_generator.to(self.device)

    #Forward propagation
    def forward(self, x):

        x = self.model_generator(x)
        x = self.out_conv(x)
        y = self.act(x)

        return y

    # Block with TransposedConv, Batch Normalization and ReLu activation function
    def transposed_conv_bn_act(self, inputs, outputs,kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(inputs, outputs, kernel_size= kernel_size, stride= stride, padding= padding),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True))



class Discriminator(torch.nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        #Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model layers
        self.layers_discriminator = [

            self.conv_bn_act(input_shape[0], 64, (4, 4), (2, 2), (1, 1)),
            self.conv_bn_act(64, 128, (4, 4), (2, 2), (1, 1)),
            self.conv_bn_act(128, 512, (4, 4), (1, 1), (1, 1)),
            self.conv_bn_act(512, 1024, (4, 4), (2, 2), (1, 1)),
            self.conv_bn_act(1024, 1024, (4, 4), (1, 1), (1, 1))
        ]

        # Out layer
        self.out_conv = nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
        self.flatten = nn.Flatten()

        # Automatic weight init
        for i in range(len(self.layers_discriminator)):
            torch.nn.init.xavier_uniform_(self.layers_discriminator[i][0].weight)
            torch.nn.init.zeros_(self.layers_discriminator[i][0].bias)

        #Create network and put on device
        self.model_discriminator = nn.Sequential(*self.layers_discriminator)
        self.model_discriminator.to(self.device)

    #Forward propagation
    def forward(self, x):

        x = self.model_discriminator(x)
        x = self.out_conv(x)
        y = self.flatten(x)

        return y

    # Block with Conv, Batch Normalization and LeakyReLU activation function
    def conv_bn_act(self, inputs, outputs, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, kernel_size= kernel_size, stride= stride, padding= padding),
            nn.BatchNorm2d(outputs),
            nn.LeakyReLU(0.2, inplace=True))
