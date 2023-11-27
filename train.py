import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import torch.optim as optim
from pathlib import Path
import torchvision

from model import Generator, Discriminator

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

torchvision.disable_beta_transforms_warning()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    print(f"Running DDP setup on rank:{rank}/{world_size}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_dataloader(cfg, dataset, device, epoch, sched, sampler):
    if cfg.ddp:
        sampler.set_epoch(epoch)
        return DataLoader(
            dataset=dataset,
            batch_size=sched.batch_size,
            sampler=sampler,
            num_workers=3,
            pin_memory=True,
            pin_memory_device=device,
            drop_last=True,
        )
    else:
        return DataLoader(
            dataset=dataset,
            batch_size=sched.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
        )


def get_checkpoint_dict(cfg, model, optimizer, fixed_noise):
    if cfg.ddp:
        model = model.module
    return {
        "model": model,
        "optimizer": optimizer,
        "fixed_noise": fixed_noise,
        "depth": model.G_net.depth,
        "alpha": model.G_net.alpha,
        "alpha_step": model.G_net.alpha_step,
    }


def get_model(cfg, model):
    if cfg.ddp:
        model = model.module
    return model


def get_sched_for_epoch(cfg, epoch):
    """Get's the schedule for a given epoch

    Args:
            epoch (int): the epoch of the run

    Returns:
            schedule: the schedule to use
    """
    idxs = [i for i, x in enumerate(cfg.schedule) if x.start_epoch <= epoch]
    return cfg.schedule[idxs[-1]] if len(idxs) > 0 else cfg.schedule[-1]


def img_transform(size):
    tf = nn.Sequential(
        transforms.Resize([size, size], antialias=True),
        transforms.CenterCrop([size, size]),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    )
    return torch.jit.script(tf)


class Wrapper(nn.Module):
    def __init__(self, G_net, D_net, latent_size, lambd):
        super().__init__()
        self.latent_size = latent_size
        self.lambd = lambd
        self.G_net = G_net
        self.D_net = D_net

    def train_G(self, samples):
        noise = torch.randn(
            samples.size(0), self.latent_size, 1, 1, device=samples.device
        )
        fake = self.G_net(noise)
        fake_out = self.D_net(fake)

        G_loss = -fake_out.mean()
        return G_loss

    def train_D(self, samples):
        noise = torch.randn(
            samples.size(0), self.latent_size, 1, 1, device=samples.device
        )
        fake = self.G_net(noise).detach()  # detach is super important here
        fake_out = self.D_net(fake)
        real_out = self.D_net(samples)

        ## Gradient Penalty
        eps = torch.rand(samples.size(0), 1, 1, 1, device=samples.device)
        eps = eps.expand_as(samples)
        x_hat = eps * samples + (1 - eps) * fake
        x_hat.requires_grad = True
        px_hat = self.D_net(x_hat)
        grad = torch.autograd.grad(
            outputs=px_hat.sum(), inputs=x_hat, create_graph=True
        )[0]
        grad_norm = grad.view(samples.size(0), -1).norm(2, dim=1)
        gradient_penalty = self.lambd * ((grad_norm - 1) ** 2).mean()

        ## Final Loss
        return fake_out.mean() - real_out.mean() + gradient_penalty

    def growing_net(self, num_iters):
        self.G_net.growing_net(num_iters)
        self.D_net.growing_net(num_iters)


def main(rank, world_size, cfg):
    """
    Main training loop. When ddp is disabled, rank is 0 and world_size is 1.
    """
    if cfg.cuda:
        torch.cuda.set_device(rank)
    device = f"cuda:{rank}" if cfg.cuda else "cpu"
    if cfg.ddp:
        setup(rank, world_size)

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

    curr_sched = get_sched_for_epoch(cfg, cfg.resume)
    num_epochs = cfg.epochs
    latent_size = cfg.latent_size
    out_res = cfg.out_res
    lr = 1e-4
    lambd = 10

    transform = transforms.ToTensor()
    model = Wrapper(
        Generator(latent_size, out_res),
        Discriminator(latent_size, out_res),
        latent_size,
        lambd,
    ).to(device)

    fixed_noise = torch.randn(16, latent_size, 1, 1, device=device)
    optimizer = optim.Adam(  # separate adam for each network
        [{"params": model.G_net.parameters()}, {"params": model.D_net.parameters()}],
        lr=lr,
        betas=(0, 0.99),
    )

    D_running_loss, G_running_loss = 0, 0
    iter_num = 0

    # Looks like checkpoint is a pickled dict with a bunch of interesting information
    if cfg.resume != 0:
        map_location = {"cuda:0": f"cuda:{rank}"}  # this is fine non-ddp too
        check_point = torch.load(
            check_point_dir / f"check_point_epoch_{cfg.resume}.pth",
            map_location=map_location,
        )  # Expects per epoch saves in a given location
        fixed_noise = check_point["fixed_noise"]
        model.G_net.load_state_dict(check_point["G_net"])
        model.D_net.load_state_dict(check_point["D_net"])
        optimizer.load_state_dict(check_point["optimizer"])
        model.G_net.depth = check_point["depth"]
        model.D_net.depth = check_point["depth"]
        model.G_net.alpha = check_point["alpha"]
        model.D_net.alpha = check_point["alpha"]
        model.G_net.alpha_step = check_point["alpha_step"]
        model.D_net.alpha_step = check_point["alpha_step"]

    if cfg.ddp:  # Do this _after_ loading the checkpoint from resume
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    sampler = (
        DistributedSampler(dataset, rank=rank, drop_last=False) if cfg.ddp else None
    )
    data_loader = get_dataloader(
        cfg, dataset, device, 1 + cfg.resume, curr_sched, sampler
    )
    log_every = max(1, len(data_loader) // 100)  # log 100 times per epoch
    size = 2 ** (get_model(cfg, model).G_net.depth + 1)
    if rank == 0:
        print(f"Output Resolution: {size}x{size}")

    for epoch in range(1 + cfg.resume, cfg.epochs + 1):
        model.train()
        curr_sched = get_sched_for_epoch(cfg, epoch)
        if rank == 0:
            print(f"epoch {epoch}/{num_epochs} schedule: {curr_sched}")
        data_loader = get_dataloader(cfg, dataset, device, epoch, curr_sched, sampler)
        log_every = max(1, len(data_loader) // 100)  # log 100 times per epoch
        if (
            epoch == curr_sched.start_epoch
            and 2 ** (get_model(cfg, model).G_net.depth + 1) < out_res
        ):
            # if this epoch is the start of a schedule
            # and increasing depth is still less than the output size
            if curr_sched.grow_epochs:  # this should be none for the first schedule
                assert epoch != 1, "The first epoch should have no grow_epochs"
                get_model(cfg, model).growing_net(
                    curr_sched.grow_epochs * len(data_loader)
                )
                size = 2 ** (get_model(cfg, model).G_net.depth + 1)
            else:
                assert epoch == 1, "Only the first schedule should have no grow_epochs"
            if rank == 0 and epoch != 1:
                print(f"Output Resolution: {size}x{size}")

        prep_images = img_transform(size)
        databar = tqdm(data_loader) if rank == 0 else data_loader
        for i, samples in enumerate(databar):
            samples = prep_images(samples[0].to(device))

            ##  update D
            model.zero_grad()
            D_loss = get_model(cfg, model).train_D(samples)
            D_loss.backward()
            get_model(cfg, model).G_net.zero_grad()  # just in case, no grads for G
            optimizer.step()
            ## update G
            model.zero_grad()
            G_loss = get_model(cfg, model).train_G(samples)
            G_loss.backward()
            optimizer.step()

            D_running_loss += D_loss.detach()
            G_running_loss += G_loss.detach()

            iter_num += 1
            if i % log_every == 0 and rank == 0:
                # if cfg.ddp:
                # dist.reduce(D_running_loss, dst=0, op=ReduceOp.SUM)
                # dist.reduce(G_running_loss, dst=0, op=ReduceOp.SUM)
                d_loss = D_running_loss.item() / (iter_num)  # * world_size)
                g_loss = G_running_loss.item() / (iter_num)  # * world_size)
                databar.set_postfix(
                    {
                        "d_loss": f"{d_loss:.3f}",
                        "g_loss": f"{g_loss:.3f}",
                    }
                )
                iter_num = 0
                D_running_loss, G_running_loss = 0.0, 0.0

        if rank == 0:
            check_point = get_checkpoint_dict(cfg, model, optimizer, fixed_noise)
            with torch.no_grad():
                model.eval()
                torch.save(
                    check_point, check_point_dir / f"check_point_epoch_{epoch}.pth"
                )
                torch.save(
                    get_model(cfg, model).state_dict(),
                    weight_dir / f"model_weight_epoch_{epoch}.pth",
                )
                out_imgs = get_model(cfg, model).G_net(fixed_noise)
                out_grid = make_grid(
                    out_imgs,
                    normalize=True,
                    nrow=4,
                    scale_each=True,
                    padding=int(0.5 * (2 ** get_model(cfg, model).G_net.depth)),
                ).permute(1, 2, 0)
                plt.imshow(out_grid.cpu())
                plt.savefig(output_dir / f"size_{size}_epoch_{epoch}")

    # Done training, clean up
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    cfg = OmegaConf.load("config.yaml")
    cfg.merge_with_cli()
    if cfg.ddp:
        mp.spawn(main, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        main(0, 1, cfg)
