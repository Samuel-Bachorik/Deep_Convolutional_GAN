import matplotlib.pyplot as plt

#Arithmetic mean
class CreateGraph:
    def __init__(self, batch_count,name):

        self.batch_count = batch_count
        self.num_for_G = 0
        self.num_for_D = 0
        self.array_epoch, self.array_loss_G, self.array_loss_D = [], [], []
        self.name = name

    def count(self, epoch):

        self.array_epoch.append(epoch)

        self.array_loss_G.append(self.num_for_G / self.batch_count)
        self.array_loss_D.append(self.num_for_D / self.batch_count)

        #print("Epoch average loss: Generator - {}, Discriminator - {}".format(self.num_for_G / self.batch_count, self.num_for_D / self.batch_count))

        self.num_for_G = 0
        self.num_for_D = 0


        plt.plot(self.array_epoch, self.array_loss_G)
        plt.plot(self.array_epoch, self.array_loss_D)

        plt.title(self.name)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Generator loss", "Discriminator loss"])
        plt.savefig(self.name + ".png")
        plt.show()
