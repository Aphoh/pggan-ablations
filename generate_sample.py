import argparse
import os
import torch
from torchvision.utils import save_image, make_grid
from omegaconf import OmegaConf
from train import Wrapper
from model import Generator, Discriminator
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="Seed for generate images")
parser.add_argument(
    "--out_dir", type=str, default="./results", help="Directory for the output images"
)
parser.add_argument(
    "--num_imgs", type=int, default=1, help="Number of images to generate"
)
parser.add_argument("--weight", type=str, help="Generator weight")
parser.add_argument("--ckpt", type=str, help="Checkpoint")
parser.add_argument(
    "--out_res", type=int, default=128, help="The resolution of final output image"
)
parser.add_argument("--cuda", action="store_true", help="Using GPU to train")

opt = parser.parse_args()

cfg = OmegaConf.load("config.yaml")

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.cuda) else "cpu")
latent_size = 512

model = Wrapper(
    Generator(latent_size, opt.out_res),
    Discriminator(latent_size, opt.out_res),
    latent_size,
    cfg.lambd,
).to(device)
check_point = torch.load(opt.ckpt, map_location=device)
model.load_state_dict(check_point["model"])
model.G_net.depth = check_point["depth"]
model.G_net.alpha = check_point["alpha"]

# noise = torch.randn(opt.num_imgs, latent_size, 1, 1, device=device)
noise = check_point["fixed_noise"]
model.eval()
out_imgs = model.G_net(noise)
print(out_imgs.shape)


def channel_norm(x):
    norm = lambda x: (x - x.min()) / (x.max() - x.min())
    return torch.stack([norm(x[:, i]) for i in range(x.shape[1])], dim=1)


plt.hist(out_imgs[:, 2].cpu().detach().numpy().flatten())

save_image(channel_norm(out_imgs), "out_grid.png")
plt.show()
