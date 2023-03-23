import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.out_channels = out_channels

        self.generator = nn.Sequential(
            nn.Linear(self.latent_dim + 10, 256*2*2),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, 2, 2)),
            # nn.ConvTranspose2d(self.latent_dim + 10, 256, 4, 2, 1),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(32, self.out_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z, y_in):
        y_in = F.one_hot(y_in, self.num_classes).to(device)
        z = torch.concat((z, y_in), dim=1)
        return self.generator(z)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.discriminator = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        self.score = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1),
            nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 10),
            nn.LogSoftmax())

    def forward(self, x):
        xd = self.discriminator(x)
        return self.score(xd), self.classifier(xd)
        

# -----
# Hyperparameters
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# NOTE: Feel free to change the hyperparameters as long as you meet the marking requirement
batch_size = 256
workers = 0
latent_dim = 128
lr = 0.001
num_epochs = 150
validate_every = 1
print_every = 100

save_path = os.path.join(os.path.curdir, "visualize", "gan")
if not os.path.exists(os.path.join(os.path.curdir, "visualize", "gan")):
    os.makedirs(os.path.join(os.path.curdir, "visualize", "gan"))
ckpt_path = 'acgan.pt'

# -----
# Dataset
# NOTE: Data is only normalized to [0, 1]. THIS IS IMPORTANT!!!
tfms = transforms.Compose([
    transforms.ToTensor(), 
    ])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True,
    transform=tfms)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=workers)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True,
    transform=tfms)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size,
    shuffle=False, 
    num_workers=workers)


# -----
# Model
generator = Generator(latent_dim, 3, 10)
generator.apply(weights_init)
discriminator = Discriminator(3)
discriminator.apply(weights_init)

# -----
# Losses

adv_loss = nn.BCELoss()
aux_loss = nn.NLLLoss()

if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    adv_loss = adv_loss.cuda()
    aux_loss = aux_loss.cuda()

# Optimizers for Discriminator and Generator, separate
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
optimizer_G = optim.Adam(generator.parameters(), lr=lr)


# -----
# Train loop

def denormalize(x):
    """Denomalize a normalized image back to uint8.
    """
    x = x.permute((0, 2,3,1)) * 255
    x = x.detach().cpu().numpy().astype(np.uint8)
    return x


# For visualization
# Generate 20 random sample for visualization
random_z = torch.randn((20, latent_dim)).to(device)
random_y = torch.randint(low=0, high=10, size=(20, )).to(device)


def compute_acc(preds, labels):
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc


def train_step(x, y):
    """One train step for AC-GAN."""
    fake_z = torch.randn((x.shape[0], latent_dim)).to(device)
    fake_y = torch.randint(low=0, high=10, size=(x.shape[0], )).to(device)

    optimizer_D.zero_grad()
    disc, cl = discriminator(x)
    real_label = torch.full((x.shape[0], ), 1).to(device).to(torch.float32)
    disc_err = adv_loss(disc.squeeze(), real_label)
    class_err = aux_loss(cl, y)
    err_D_real = disc_err + class_err

    err_D_real.backward()
    optimizer_D.step()

    g_im = generator(fake_z, fake_y)
    disc, cl_g = discriminator(g_im.detach())
    fake_label = torch.full((x.shape[0],), 0).to(device).to(torch.float32)
    disc_G_err = adv_loss(disc.squeeze(), fake_label)
    class_D_err = aux_loss(cl_g, fake_y)
    err_D_fake = disc_G_err + class_D_err
    total = err_D_fake + err_D_real

    acc = compute_acc(torch.concat((cl, cl_g)), torch.concat((y, fake_y)))

    err_D_fake.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()
    disc_G, cl_G = discriminator(g_im)
    gen_err = adv_loss(disc_G.squeeze(), real_label)
    class_G_err = aux_loss(cl_G, y)
    error_fake = gen_err + class_G_err

    error_fake.backward()
    optimizer_G.step()

    return error_fake, total, acc

def test(
    test_loader,
    ):
    """Calculate accuracy over Cifar10 test set.
    """
    size = len(test_loader.dataset)
    corrects = 0

    discriminator.eval()
    with torch.no_grad():
        for inputs, gts in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                gts = gts.cuda()

            _, outputs = discriminator(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == gts.data)

    acc = corrects / size
    print("Test Acc: {:.4f}".format(acc))
    return acc


g_losses = []
d_losses = []
best_acc_test = 0.0

for epoch in range(1, num_epochs + 1):
    generator.train()
    discriminator.train()

    avg_loss_g, avg_loss_d = 0.0, 0.0
    for i, (x, y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        # train step
        loss_g, loss_d, acc_d = train_step(x, y)
        avg_loss_g += loss_g * x.shape[0]
        avg_loss_d += loss_d * x.shape[0]

        # Print
        if i % print_every == 0:
            print("Epoch {}, Iter {}: LossG: {:.6f} LossD: {:.6f}, D_acc {:.6f}".format(epoch, i, loss_g, loss_d, acc_d))

    g_losses.append(avg_loss_g.item() / len(train_dataset))
    d_losses.append(avg_loss_d.item() / len(train_dataset))

    # Save
    if epoch % validate_every == 0:
        acc_test = test(test_loader)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            # Wrap things to a single dict to train multiple model weights
            state_dict = {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                }
            torch.save(state_dict, ckpt_path)
            print("Best model saved w/ Test Acc of {:.6f}.".format(best_acc_test))


        # Do some reconstruction
        generator.eval()
        with torch.no_grad():
            # Forward
            xg = generator(random_z, random_y)
            xg = denormalize(xg)

            # Plot 20 randomly generated images
            plt.figure(figsize=(10, 5))
            for p in range(20):
                plt.subplot(4, 5, p+1)
                plt.imshow(xg[p])
                plt.text(0, 0, "{}".format(classes[random_y[p].item()]), color='black',
                            backgroundcolor='white', fontsize=8)
                plt.axis('off')

            plt.savefig(os.path.join(os.path.join(save_path, "E{:d}.png".format(epoch))), dpi=300)
            plt.clf()
            plt.close('all')

        # Plot losses
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(g_losses, label="G")
        plt.plot(d_losses, label="D")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim([1, epoch])
        plt.legend()
        plt.savefig(os.path.join(os.path.join(save_path, "loss.png")), dpi=300)