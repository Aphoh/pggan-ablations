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

import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
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
            num_workers=0,
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


def get_checkpoint_dict(cfg, G_net, D_net, G_optimizer, D_optimizer, fixed_noise):
    if cfg.ddp:
        G_net = G_net.module
        D_net = D_net.module
    return {
        "G_net": G_net.state_dict(),
        "G_optimizer": G_optimizer.state_dict(),
        "D_net": D_net.state_dict(),
        "D_optimizer": D_optimizer.state_dict(),
        "fixed_noise": fixed_noise,
        "depth": G_net.depth,
        "alpha": G_net.alpha,
        "alpha_step": G_net.alpha_step,
    }


def get_sched_for_epoch(cfg, epoch):
    """Get's the schedule for a given epoch

    Args:
            epoch (int): the epoch of the run

    Returns:
            schedule: the schedule to use
    """
    idxs = [i for i, x in enumerate(cfg.schedule) if x.start_epoch <= epoch]
    return cfg.schedule[idxs[-1]] if len(idxs) > 0 else cfg.schedule[-1]


def main(rank, world_size, cfg):
    """
    Main training loop. When ddp is disabled, rank is 0 and world_size is 1.
    """
    print(f"Running DDP setup on rank:{rank}/{world_size}")
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

    transform = transforms.Compose(
        [
            transforms.Resize(out_res),
            transforms.CenterCrop(out_res),
            transforms.ToTensor(),
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

    # Looks like checkpoint is a pickled dict with a bunch of interesting information
    if cfg.resume != 0:
        map_location = {"cuda:0": f"cuda:{rank}"}  # this is fine non-ddp too
        check_point = torch.load(
            check_point_dir / f"check_point_epoch_{cfg.resume}.pth",
            map_location=map_location,
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

    if cfg.ddp:  # Do this _after_ loading the checkpoint from resume
        D_net = DDP(D_net, device_ids=[rank], find_unused_parameters=True)
        G_net = DDP(G_net, device_ids=[rank], find_unused_parameters=True)

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    sampler = (
        DistributedSampler(dataset, rank=rank, drop_last=False) if cfg.ddp else None
    )
    data_loader = get_dataloader(
        cfg, dataset, device, 1 + cfg.resume, curr_sched, sampler
    )
    log_every = max(1, len(data_loader) // 100)  # log 100 times per epoch
    size = 2 ** (G_net.depth + 1)
    print(f"Output Resolution: {size}x{size}")

    for epoch in range(1 + cfg.resume, cfg.epochs + 1):
        G_net.train()
        curr_sched = get_sched_for_epoch(cfg, epoch)
        print(f"epoch {epoch}/{num_epochs} schedule: {curr_sched}")
        data_loader = get_dataloader(cfg, dataset, device, epoch, curr_sched, sampler)
        log_every = max(1, len(data_loader) // 100)  # log 100 times per epoch
        if epoch == curr_sched.start_epoch and 2 ** (G_net.depth + 1) < out_res:
            # if this epoch is the start of a schedule
            # and increasing depth is still less than the output size
            if curr_sched.grow_epochs:  # this should be none for the first schedule
                assert epoch != 1, "The first epoch should have no grow_epochs"
                G_net.growing_net(curr_sched.grow_epochs * len(data_loader))
                D_net.growing_net(curr_sched.grow_epochs * len(data_loader))
                size = 2 ** (G_net.depth + 1)
            else:
                assert epoch == 1, "Only the first schedule should have no grow_epochs"
            print(f"Output Resolution: {size}x{size}")

        databar = tqdm(data_loader)
        for i, samples in enumerate(databar):
            ##  update D
            samples = samples[0].to(device)
            if size != out_res:
                samples = F.interpolate(samples[0], size=size)
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

            if i % log_every == 0 and rank == 0:
                if cfg.ddp:
                    D_running_loss = dist.reduce(D_running_loss, dst=0, op=ReduceOp.AVG)
                    G_running_loss = dist.reduce(G_running_loss, dst=0, op=ReduceOp.AVG)
                D_running_loss /= iter_num
                G_running_loss /= iter_num
                databar.set_postfix(
                    {
                        "d_loss": f"{D_running_loss:.3f}",
                        "g_loss": f"{G_running_loss:.3f}",
                    }
                )
                iter_num = 0
                D_running_loss, G_running_loss = 0.0, 0.0

        if rank == 0:
            check_point = get_checkpoint_dict(
                cfg, G_net, D_net, G_optimizer, D_optimizer, fixed_noise
            )
            with torch.no_grad():
                G_net.eval()
                torch.save(
                    check_point, check_point_dir / f"check_point_epoch_{epoch}.pth"
                )
                torch.save(
                    G_net.state_dict(), weight_dir / f"G_weight_epoch_{epoch}.pth"
                )
                out_imgs = G_net(fixed_noise)
                out_grid = make_grid(
                    out_imgs,
                    normalize=True,
                    nrow=4,
                    scale_each=True,
                    padding=int(0.5 * (2**G_net.depth)),
                ).permute(1, 2, 0)
                plt.imshow(out_grid.cpu())
                plt.savefig(output_dir / f"size_{size}_epoch_{epoch}")

    # Done training, clean up
    cleanup()


if __name__ == "main":
    world_size = torch.cuda.device_count()
    cfg = OmegaConf.load("config.yaml")
    cfg.merge_with_cli()
    if cfg.ddp:
        mp.spawn(main, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        main(0, 1, cfg)
