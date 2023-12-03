import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.transforms import InterpolationMode
from torch import fft

import matplotlib.pyplot as plt
import torch.optim as optim
import wandb

from model import Generator, Discriminator


def smooth_highpass_filt(img_ifft, pct):
    w, h = img_ifft.shape[-2:]
    xx, yy = torch.meshgrid(
        torch.arange(-w // 2, w // 2, device=img_ifft.device),
        torch.arange(-h // 2, h // 2, device=img_ifft.device),
        indexing="xy",
    )
    s2 = 2**0.5
    # rotate 45 to get square filter for FFT
    xx2 = xx * s2 - yy * s2
    yy2 = xx * s2 + yy * s2
    xx, yy = xx2, yy2
    rr = torch.norm(
        torch.stack((xx.to(torch.float32), yy.to(torch.float32)), dim=-1), dim=-1, p=1
    )
    rr /= rr.max()
    b = pct**3
    mask = torch.min(
        torch.ones_like(rr), (rr < b).int() + torch.exp(-((rr - b) ** 2) / 1e-4)
    )
    return mask * img_ifft


def _fft_filt(img: torch.Tensor, pct: torch.Tensor):
    img_ifft = fft.ifftshift(fft.ifft2(img))
    img_ifft = smooth_highpass_filt(img_ifft, pct)
    return fft.fft2(fft.fftshift(img_ifft)).real.clip(0, 1)


fft_filt = torch.jit.script(_fft_filt)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root", type=str, default="./", help="directory contrains the data and outputs"
)
parser.add_argument("--epochs", type=int, default=47, help="training epoch number")
parser.add_argument(
    "--out_res", type=int, default=128, help="The resolution of final output image"
)
parser.add_argument("--resume", type=int, default=0, help="continues from epoch number")
parser.add_argument("--cuda", action="store_true", help="Using GPU to train")
parser.add_argument("--output_dir", type=str, help="Output directory")
parser.add_argument("--ckpt_dir", type=str, help="Checkpoint directory")
parser.add_argument(
    "--nearest", action="store_true", help="Using nearest interpolation"
)
parser.add_argument("--fft", action="store_true", help="Using fft filter")


opt = parser.parse_args()

root = opt.root
data_dir = root + "dataset/"
check_point_dir = opt.ckpt_dir
output_dir = opt.output_dir
weight_dir = root + "weight/"
if not os.path.exists(check_point_dir):
    os.makedirs(check_point_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

## The schedule contains [num of epoches for starting each size][batch size for each size][num of epoches]
schedule = [[5, 15, 25, 35, 40, 45], [16, 16, 16, 8, 4, 2], [5, 5, 5, 1, 1, 1]]
batch_size = schedule[1][0]
growing = schedule[2][0]
epochs = opt.epochs
latent_size = 512
out_res = opt.out_res
lr = 1e-4
lambd = 10

device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.cuda) else "cpu")

pre_transform = transforms.Compose(
    [
        transforms.Resize(  # no-op if out_res is 256
            out_res,
            interpolation=InterpolationMode.NEAREST
            if opt.nearest
            else InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.CenterCrop(out_res),
        transforms.ToTensor(),
    ]
)

post_transform = transforms.Compose(
    [
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


D_net = Discriminator(latent_size, out_res).to(device)
G_net = Generator(latent_size, out_res).to(device)

fixed_noise = torch.randn(16, latent_size, 1, 1, device=device)
D_optimizer = optim.Adam(D_net.parameters(), lr=lr, betas=(0, 0.99))
G_optimizer = optim.Adam(G_net.parameters(), lr=lr, betas=(0, 0.99))


D_running_loss = 0.0
G_running_loss = 0.0
iter_num = 0

D_epoch_losses = []
G_epoch_losses = []

if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs")
    D_net = nn.DataParallel(D_net)
    G_net = nn.DataParallel(G_net)

if opt.resume != 0:
    check_point = torch.load(check_point_dir + "check_point_epoch_%i.pth" % opt.resume)
    fixed_noise = check_point["fixed_noise"]
    G_net.load_state_dict(check_point["G_net"])
    D_net.load_state_dict(check_point["D_net"])
    G_optimizer.load_state_dict(check_point["G_optimizer"])
    D_optimizer.load_state_dict(check_point["D_optimizer"])
    G_epoch_losses = check_point["G_epoch_losses"]
    D_epoch_losses = check_point["D_epoch_losses"]
    G_net.depth = check_point["depth"]
    D_net.depth = check_point["depth"]
    G_net.alpha = check_point["alpha"]
    D_net.alpha = check_point["alpha"]


try:
    c = next(x[0] for x in enumerate(schedule[0]) if x[1] > opt.resume) - 1
    batch_size = schedule[1][c]
    growing = schedule[2][c]
    dataset = datasets.ImageFolder(data_dir, transform=pre_transform)
    # dataset = datasets.CelebA(data_dir, split='all', transform=transform)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    tot_iter_num = len(dataset) / batch_size
    G_net.fade_iters = (
        (1 - G_net.alpha) / (schedule[0][c + 1] - opt.resume) / (2 * tot_iter_num)
    )
    D_net.fade_iters = (
        (1 - D_net.alpha) / (schedule[0][c + 1] - opt.resume) / (2 * tot_iter_num)
    )


except:
    print("Fully Grown\n")
    # dead code
    c = -1
    batch_size = schedule[1][c]
    growing = schedule[2][c]

    dataset = datasets.CelebA(data_dir, split="all", transform=transform)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    tot_iter_num = len(dataset) / batch_size
    print(schedule[0][c], opt.resume)

    if G_net.alpha < 1:
        G_net.fade_iters = (
            (1 - G_net.alpha) / (opt.epochs - opt.resume) / (2 * tot_iter_num)
        )
        D_net.fade_iters = (
            (1 - D_net.alpha) / (opt.epochs - opt.resume) / (2 * tot_iter_num)
        )


size = 2 ** (G_net.depth + 1)
print("Output Resolution: %d x %d" % (size, size))

wandb.init()

for epoch in range(1 + opt.resume, opt.epochs + 1):
    G_net.train()
    D_epoch_loss = 0.0
    G_epoch_loss = 0.0
    if epoch - 1 in schedule[0]:
        if 2 ** (G_net.depth + 1) < out_res:
            c = schedule[0].index(epoch - 1)
            batch_size = schedule[1][c]
            growing = schedule[2][0]
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                drop_last=True,
            )
            tot_iter_num = tot_iter_num = len(dataset) / batch_size
            G_net.growing_net(growing * tot_iter_num)
            D_net.growing_net(growing * tot_iter_num)
            size = 2 ** (G_net.depth + 1)
            print("Output Resolution: %d x %d" % (size, size))

    print("epoch: %i/%i" % (int(epoch), int(epochs)))
    databar = tqdm(data_loader)

    for i, samples in enumerate(databar):
        ##  update D
        samples = samples[0].to(device)
        if opt.fft:
            samples = fft_filt(
                samples, torch.tensor(epoch / epochs, device=samples.device)
            )
        samples = post_transform(samples)  # normalization
        if size != out_res:
            samples = F.interpolate(
                samples,
                mode="nearest" if opt.nearest else "bilinear",
                size=size,
                antialias=not opt.nearest,
            )

        D_net.zero_grad()
        noise = torch.randn(samples.size(0), latent_size, 1, 1, device=device)
        fake = G_net(noise)
        fake_out = D_net(fake.detach())
        real_out = D_net(samples)

        ## Gradient Penalty

        eps = torch.rand(samples.size(0), 1, 1, 1, device=device)
        eps = eps.expand_as(samples)
        x_hat = eps * samples + (1 - eps) * fake.detach()
        x_hat.requires_grad = True
        px_hat = D_net(x_hat)
        grad = torch.autograd.grad(
            outputs=px_hat.sum(), inputs=x_hat, create_graph=True
        )[0]
        grad_norm = grad.view(samples.size(0), -1).norm(2, dim=1)
        gradient_penalty = lambd * ((grad_norm - 1) ** 2).mean()

        ###########

        D_loss = fake_out.mean() - real_out.mean() + gradient_penalty

        if not torch.isnan(D_loss):
            D_loss.backward()
            D_optimizer.step()
        else:
            D_loss = 0

        ##	update G

        G_net.zero_grad()
        fake_out = D_net(fake)

        G_loss = -fake_out.mean()

        if not torch.isnan(G_loss):
            G_loss.backward()
            G_optimizer.step()
        else:
            G_loss = 0

        ##############

        D_running_loss += D_loss.item()
        G_running_loss += G_loss.item()

        iter_num += 1

        if i % 500 == 0:
            D_running_loss /= iter_num
            G_running_loss /= iter_num
            wandb.log(
                {
                    "D_loss": D_running_loss,
                    "G_loss": G_running_loss,
                    "gp": gradient_penalty,
                    "epoch": epoch,
                    "size": size,
                }
            )
            databar.set_description(
                "D_loss: %.3f   G_loss: %.3f, gp: %.2f"
                % (D_running_loss, G_running_loss, gradient_penalty)
            )
            iter_num = 0
            D_running_loss = 0.0
            G_running_loss = 0.0

    D_epoch_losses.append(D_epoch_loss / tot_iter_num)
    G_epoch_losses.append(G_epoch_loss / tot_iter_num)

    with torch.no_grad():
        G_net.eval()
        # save checkpoint after each size change
        if epoch + 1 in schedule[0] or epoch == opt.epochs:
            check_point = {
                "G_net": G_net.state_dict(),
                "G_optimizer": G_optimizer.state_dict(),
                "D_net": D_net.state_dict(),
                "D_optimizer": D_optimizer.state_dict(),
                "D_epoch_losses": D_epoch_losses,
                "G_epoch_losses": G_epoch_losses,
                "fixed_noise": fixed_noise,
                "depth": G_net.depth,
                "alpha": G_net.alpha,
            }
            checkpoint_loc = check_point_dir + "check_point_epoch_%d.pth" % (epoch)
            torch.save(check_point, checkpoint_loc)
            artifact = wandb.Artifact(
                type="checkpoint",
                name="checkpoint_size_%d_epoch_%d" % (size, epoch),
                description="checkpoint for size %d epoch %d" % (size, epoch),
            )
            artifact.add_file(checkpoint_loc)
            wandb.log_artifact(artifact)

        out_imgs = G_net(fixed_noise)
        out_grid = make_grid(
            out_imgs,
            normalize=True,
            nrow=4,
            scale_each=True,
            padding=int(0.5 * (2**G_net.depth)),
        ).permute(1, 2, 0)
        plt.imshow(out_grid.cpu())
        plt.savefig(output_dir + "size_%i_epoch_%d" % (size, epoch))
        img = wandb.Image(out_grid.cpu().numpy())
        wandb.log({"generated_images": img, "epoch": epoch})
