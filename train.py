import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import torch.optim as optim
from pathlib import Path

from model import Generator, Discriminator

cfg = OmegaConf.load("config.yaml").merge_with_cli()

root = Path(".")
data_dir = root / cfg.data_dir
check_point_dir = root / cfg.checkpoint_dir
output_dir = root / cfg.output_dir
weight_dir = root / cfg.weight_dir
if not os.path.exists(check_point_dir):
    os.makedirs(check_point_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

# schedule = [[5, 15, 25, 35, 40], [16, 16, 16, 8, 4], [5, 5, 5, 1, 1]]
sched_start_epochs = [sched.start_epoch for sched in cfg.schedule]
sched_batch_sizes = [sched.batch_size for sched in cfg.schedule]
sched_grow_epochs = [sched.grow_epochs for sched in cfg.schedule]


def get_sched_for_epoch(epoch):
    """Get's the schedule for a given epoch

    Args:
            epoch (int): the epoch of the run

    Returns:
            schedule: the schedule to use
    """
    return [x for x in cfg.schedule if x.start_epoch <= epoch].get(-1, cfg.schedule[-1])


curr_sched = get_sched_for_epoch(cfg.resume)
num_epochs = cfg.epochs
latent_size = cfg.latent_size
out_res = cfg.out_res
lr = 1e-4
lambd = 10

# TODO: DDP here
device = torch.device("cuda:0" if (torch.cuda.is_available() and cfg.cuda) else "cpu")
transform = transforms.Compose(
    [
        transforms.Resize(out_res),
        transforms.CenterCrop(out_res),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

## TODO: DDP Wrap
D_net = Discriminator(latent_size, out_res).to(device)
G_net = Generator(latent_size, out_res).to(device)

fixed_noise = torch.randn(16, latent_size, 1, 1, device=device)
D_optimizer = optim.Adam(D_net.parameters(), lr=lr, betas=(0, 0.99))
G_optimizer = optim.Adam(G_net.parameters(), lr=lr, betas=(0, 0.99))


D_running_loss = 0.0
G_running_loss = 0.0
iter_num = 0

if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs")
    # TODO: DDP
    D_net = nn.DataParallel(D_net)
    G_net = nn.DataParallel(G_net)

# Looks like checkpoint is a pickled dict with a bunch of interesting information
if cfg.resume != 0:
    check_point = torch.load(
        check_point_dir + "check_point_epoch_%i.pth" % cfg.resume
    )  # Expects per epoch saves in a given location
    fixed_noise = check_point["fixed_noise"]
    G_net.load_state_dict(check_point["G_net"])
    D_net.load_state_dict(check_point["D_net"])
    G_optimizer.load_state_dict(check_point["G_optimizer"])
    D_optimizer.load_state_dict(check_point["D_optimizer"])
    G_net.depth = check_point["depth"]
    D_net.depth = check_point["depth"]
    G_net.alpha = check_point["alpha"]
    D_net.alpha = check_point["alpha"]
    G_net.alpha_step = check_point["alpha_step"]
    D_net.alpha_step = check_point["alpha_step"]


dataset = datasets.ImageFolder(data_dir, transform=transform)
# dataset = datasets.CelebA(data_dir, split='all', transform=transform)
data_loader = DataLoader(  # TODO: DDP shuffler
    dataset=dataset,
    batch_size=curr_sched.batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=True,
)
tot_iter_num = len(dataset) / curr_sched.batch_size  # Num of iters per epoch

size = 2 ** (G_net.depth + 1)
print("Output Resolution: %d x %d" % (size, size))

for epoch in range(1 + cfg.resume, cfg.epochs + 1):
    G_net.train()
    curr_sched = get_sched_for_epoch(epoch)
    if epoch == curr_sched.start_epoch:  # if this epoch is the start of a schedule
        # and increasing depth is still less than the output size
        if 2 ** (G_net.depth + 1) < out_res:
            # TODO: account for multiple machines in the total batch size
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=curr_sched.batch_size,
                shuffle=True,
                num_workers=8,
                drop_last=True,
            )  # Why is dataloader redefined here? -> to change the batch size
            tot_iter_num = len(dataset) / curr_sched.batch_size
            G_net.growing_net(curr_sched.growing_epochs * tot_iter_num)
            D_net.growing_net(curr_sched.growing_epochs * tot_iter_num)
            size = 2 ** (G_net.depth + 1)
            print("Output Resolution: %d x %d" % (size, size))

    print("epoch: %i/%i" % (int(epoch), int(num_epochs)))
    databar = tqdm(data_loader)

    for i, samples in enumerate(databar):
        ##  update D
        if size != out_res:
            samples = F.interpolate(samples[0], size=size).to(device)
        else:
            samples = samples[0].to(device)
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

        D_loss.backward()
        D_optimizer.step()

        ##	update G

        G_net.zero_grad()
        fake_out = D_net(fake)

        G_loss = -fake_out.mean()

        G_loss.backward()
        G_optimizer.step()

        ##############

        D_running_loss += D_loss.item()
        G_running_loss += G_loss.item()

        iter_num += 1

        if i % 500 == 0:
            D_running_loss /= iter_num
            G_running_loss /= iter_num
            databar.set_postfix(
                {"d_loss": f"{D_running_loss:.3f}", "g_loss": f"{G_running_loss:.3f}"}
            )
            iter_num = 0
            D_running_loss, G_running_loss = 0.0, 0.0

    # TODO: only save if we're the master node in ddp
    check_point = {
        "G_net": G_net.state_dict(),
        "G_optimizer": G_optimizer.state_dict(),
        "D_net": D_net.state_dict(),
        "D_optimizer": D_optimizer.state_dict(),
        "fixed_noise": fixed_noise,
        "depth": G_net.depth,
        "alpha": G_net.alpha,
        "alpha_step": G_net.alpha_step,
    }
    with torch.no_grad():
        G_net.eval()
        torch.save(check_point, check_point_dir + "check_point_epoch_%d.pth" % (epoch))
        torch.save(G_net.state_dict(), weight_dir + "G_weight_epoch_%d.pth" % (epoch))
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
