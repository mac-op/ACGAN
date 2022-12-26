import os
import time
from math import ceil

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

from ssim import SSIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----
# VAE Build Blocks


class Encoder(nn.Module):
    def __init__(
            self,
            latent_dim: int = 128,
            in_channels: int = 3,
    ):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(self.in_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(6, 10, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(10, 12, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 15, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(15, 18, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(18*4*4, self.latent_dim)
        self.fc2 = nn.Linear(18*4*4, self.latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc1(x)
        log_var = self.fc2(x)

        return mu, log_var


class Decoder(nn.Module):
    def __init__(
            self,
            latent_dim: int = 128,
            out_channels: int = 3,
    ):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.decoder = nn.Sequential(
            nn.BatchNorm1d(self.latent_dim),
            nn.Linear(self.latent_dim, 18*4*4),
            nn.ReLU(),
            nn.Unflatten(1, (18, 4, 4)),
            nn.ConvTranspose2d(18, 9, kernel_size=5, stride=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(9, 3, kernel_size=5, stride=2,padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)


# #####
# Wrapper for Variational Autoencoder
# #####

class VAE(nn.Module):
    def __init__(
            self,
            latent_dim: int = 128,
    ):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encode = Encoder(latent_dim=latent_dim)
        self.decode = Decoder(latent_dim=latent_dim)

    def reparameterize(self, mu, log_var):
        """Reparameterization Tricks to sample latent vector z
        from distribution w/ mean and variance.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x, y):
        """Forward for CVAE.
        Returns:
            xg: reconstructed image from decoder.
            mu, log_var: mean and log(std) of z ~ N(mu, sigma^2)
            z: latent vector, z = mu + sigma * eps, acquired from reparameterization trick. 
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        xg = self.decode(z)
        return xg, mu, log_var, z

    def generate(
            self,
            n_samples: int,
    ):
        """Randomly sample from the latent space and return
        the reconstructed samples.
        Returns:
            xg: reconstructed image
            None: a placeholder simply.
        """
        x_in = torch.randn((n_samples, self.latent_dim)).to(device)
        xg = self.decode(x_in)

        return xg, None


# #####
# Wrapper for Conditional Variational Autoencoder
# #####
class CVAE(nn.Module):
    def __init__(
            self,
            latent_dim: int = 128,
            num_classes: int = 10,
            img_size: int = 32,
    ):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size

        self.encode = Encoder(latent_dim=latent_dim, in_channels=3)
        self.decode = Decoder(latent_dim=latent_dim+num_classes)

    def reparameterize(self, mu, log_var):
        """Reparameterization Tricks to sample latent vector z
        from distribution w/ mean and variance.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x, y):
        """Forward for CVAE.
        Returns:
            xg: reconstructed image from decoder.
            mu, log_var: mean and log(std) of z ~ N(mu, sigma^2)
            z: latent vector, z = mu + sigma * eps, acquired from reparameterization trick. 
        """
        y = F.one_hot(y, self.num_classes)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)
        z = torch.concat((z, y), -1)
        xg = self.decode(z)
        return xg, mu, log_var, z

    def generate(
            self,
            n_samples: int,
            y: torch.Tensor = None,
    ):
        """Randomly sample from the latent space and return
        the reconstructed samples.
        NOTE: Randomly generate some classes here, if not y is provided.
        Returns:
            xg: reconstructed image
            y: classes for xg. 
        """
        x = torch.randn((n_samples, self.latent_dim)).to(device)
        if not y:
            y = torch.randint(low=0, high=self.num_classes, size=(n_samples, ))
        y_one_hot = F.one_hot(y, self.num_classes).to(device)
        z = torch.concat((x, y_one_hot), -1)
        xg = self.decode(z)
        return xg, y


# #####
# Wrapper for KL Divergence
# #####
class KLDivLoss(nn.Module):
    def __init__(
            self,
            lambd: float = 1.0,
    ):
        super(KLDivLoss, self).__init__()
        self.lambd = lambd

    def forward(
            self,
            mu,
            log_var,
    ):
        loss = 0.5 * torch.sum(-log_var - 1 + mu ** 2 + log_var.exp(), dim=1)
        return self.lambd * torch.mean(loss)


# -----
# Hyperparameters
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# NOTE: Feel free to change the hyperparameters as long as you meet the marking requirement
# NOTE: DO NOT TRAIN IT LONGER THAN 100 EPOCHS.
batch_size = 256
workers = 0
latent_dim = 128
lr = 0.001
num_epochs = 60
validate_every = 1
print_every = 100

conditional = False  # Flag to use VAE or CVAE

if conditional:
    name = "cvae"
else:
    name = "vae"

# Set up save paths
if not os.path.exists(os.path.join(os.path.curdir, "visualize", name)):
    os.makedirs(os.path.join(os.path.curdir, "visualize", name))
save_path = os.path.join(os.path.curdir, "visualize", name)
ckpt_path = name + '.pt'

steps = num_epochs//4
kl_annealing = [0.] * steps + [1e-5] * steps + [1e-4] * steps + [1e-3] * steps  # KL Annealing

# -----
# Dataset
# NOTE: Data is only normalized to [0, 1]. THIS IS IMPORTANT!!!

# -----
# Model
if conditional:
    model = CVAE(latent_dim=latent_dim)
else:
    model = VAE(latent_dim=latent_dim)


bceloss = nn.BCELoss()
l2loss = nn.MSELoss()
ssimloss = SSIM()          # 1 - ssim
kld_loss = KLDivLoss(0)

best_total_loss = float("inf")

# Send to GPU
if torch.cuda.is_available():
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)

# To further help with training
# NOTE: You can remove this if you find this unhelpful
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, [40, 50], gamma=0.1, verbose=False)

# -----
# Train loop


def train_step(x, y):
    """One train step for VAE/CVAE.
    Args:
        x, y: one batch (images, labels) from Cifar10 train set.
    Returns:
        loss: total loss per batch.
        l2_loss: MSE loss for reconstruction.
        bce_loss: binary cross-entropy loss for reconstruction.
        ssim_loss: ssim loss for reconstruction.
        kldiv_loss: kl divergence loss.
    """
    x_g, mu, log_var, _ = model(x, y)

    l2 = l2loss(x_g, x)
    bce = bceloss(x_g, x)
    ssim = (1 - ssimloss(x_g, x))
    kldiv = kld_loss(mu, log_var)
    total = l2 + bce + ssim + kldiv

    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    return total, l2, bce, ssim, kldiv


def denormalize(x):
    """Denomalize a normalized image back to uint8.
    Args:
        x: torch.Tensor, in [0, 1].
    Return:
        x_denormalized: denormalized image as numpy.uint8, in [0, 255].
    """
    x = x.permute((0, 2,3,1)) * 255
    x = x.detach().cpu().numpy().astype(np.uint8)
    return x

if __name__ == "__main__":
# Loop HERE
    tfms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=tfms)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=tfms,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers)

    subset = torch.utils.data.Subset(
        test_dataset,
        [0, 380, 500, 728, 1000, 2300, 3400, 4300, 4800, 5000])

    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=10)

    l2_losses = []
    bce_losses = []
    ssim_losses = []
    kld_losses = []
    total_losses = []

    total_losses_train = []

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        total_loss_train = 0.0
        l2_losses_train = 0.0
        bce_losses_train = 0.0
        ssim_losses_train = 0.0
        kld_losses_train = 0.0

        for i, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # Train step
            model.train()
            loss, recon_loss, bce_loss, ssim_loss, kldiv_loss = train_step(x, y)
            total_loss_train += loss.item() * batch_size
            l2_losses_train += recon_loss.item() * batch_size
            bce_losses_train += bce_loss.item() * batch_size
            ssim_losses_train += ssim_loss.item() * batch_size
            kld_losses_train += kldiv_loss.item() * batch_size

            # Print
            if i % print_every == 0:
                print("Epoch {}, Iter {}: Total Loss: {:.6f} MSE: {:.6f}, SSIM: {:.6f}, BCE: {:.6f}, KLDiv: {:.6f}".format(epoch, i, loss, recon_loss, ssim_loss, bce_loss, kldiv_loss))

        total_losses_train.append(total_loss_train / len(train_dataset))
        l2_losses.append(l2_losses_train / len(train_dataset))
        bce_losses.append(bce_losses_train / len(train_dataset))
        ssim_losses.append(ssim_losses_train / len(train_dataset))
        kld_losses.append(kld_losses_train / len(train_dataset))

        # Test loop
        if epoch % validate_every == 0:
            # Loop through test set
            model.eval()

            avg_total_recon_loss_test = 0
            with torch.no_grad():
                for x, y in test_loader:
                    if torch.cuda.is_available():
                        x = x.cuda()
                        y = y.cuda()

                    xg, mu, log_var, _ = model(x, y)
                    l2 = l2loss(xg, x)
                    bce = bceloss(xg, x)
                    ssim = (1 - ssimloss(xg, x))
                    kldiv = kld_loss(mu, log_var)
                    total = l2 + bce + ssim + kldiv

                    avg_total_recon_loss_test += total.item()

                avg_total_recon_loss_test /= len(test_loader)
                total_losses.append(avg_total_recon_loss_test)

                # Plot losses
                if epoch > 1:
                    plt.plot(l2_losses, label="L2 Reconstruction")
                    plt.plot(bce_losses, label="BCE")
                    plt.plot(ssim_losses, label="SSIM")
                    plt.plot(kld_losses, label="KL Divergence")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.xlim([1, epoch])
                    plt.legend()
                    plt.savefig(os.path.join(os.path.join(save_path, "losses.png")), dpi=300)
                    plt.clf()
                    plt.close('all')

                    plt.plot(total_losses, label="Total Loss Test")
                    plt.plot(total_losses_train, label="Total Loss Train")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.xlim([1, epoch])
                    plt.legend()
                    plt.savefig(os.path.join(os.path.join(save_path, "total_loss.png")), dpi=300)
                    plt.clf()
                    plt.close('all')

                # Save best model
                if avg_total_recon_loss_test < best_total_loss:
                    torch.save(model.state_dict(), ckpt_path)
                    best_total_loss = avg_total_recon_loss_test
                    print("Best model saved w/ Total Reconstruction Loss of {:.6f}.".format(best_total_loss))

            # Do some reconstruction
            model.eval()
            with torch.no_grad():
                x, y = next(iter(loader))
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                # y_onehot = F.one_hot(y, 10).float()
                xg, _, _, _ = model(x, y)

                # Visualize
                xg = denormalize(xg)
                x = denormalize(x)

                y = y.cpu().numpy()

                plt.figure(figsize=(10, 5))
                for p in range(10):
                    plt.subplot(4, 5, p + 1)
                    plt.imshow(xg[p])
                    plt.subplot(4, 5, p + 1 + 10)
                    plt.imshow(x[p])
                    plt.text(0, 0, "{}".format(classes[y[p].item()]), color='black',
                             backgroundcolor='white', fontsize=8)
                    plt.axis('off')

                plt.savefig(os.path.join(os.path.join(save_path, "E{:d}.png".format(epoch))), dpi=300)
                plt.clf()
                plt.close('all')
                print("Figure saved at epoch {}.".format(epoch))

        kld_loss.lambd = kl_annealing[(epoch - 1)]

        print("Lambda:", kld_loss.lambd)
        end = time.time()

        # LR decay
        scheduler.step()

        print(end-start)
        print()

    # Generate some random samples
    if conditional:
        model = CVAE(latent_dim=latent_dim)
    else:
        model = VAE(latent_dim=latent_dim)
    if torch.cuda.is_available():
        model = model.cuda()
    ckpt = torch.load(name + '.pt')
    model.load_state_dict(ckpt)

    # Generate 20 random images
    xg, y = model.generate(20)
    xg = denormalize(xg)
    if y is not None:
        y = y.cpu().numpy()

    plt.figure(figsize=(10, 5))
    for p in range(20):
        plt.subplot(4, 5, p + 1)
        if y is not None:
            plt.text(0, 0, "{}".format(classes[y[p].item()]), color='black',
                     backgroundcolor='white', fontsize=8)
        plt.imshow(xg[p])
        plt.axis('off')

    plt.savefig(os.path.join(os.path.join(save_path, "random.png")), dpi=300)
    plt.clf()
    plt.close('all')
