import argparse
import os

import torch
import torchvision
import wandb
from torch.nn import functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from viewmaker.src.datasets import datasets
from viewmaker.src.gans.calc_cifar_fid_stats import calculate_cifar_fid_stats
from viewmaker.src.gans.calc_generator_fid import calculate_generator_fid
from viewmaker.src.gans.tiny_pix2pix import TinyP2PGenerator, TinyP2PDiscriminator

def train_tiny_pix2pix(args):

    G = TinyP2PGenerator(learn_residual=False, in_channels=args.in_channels).cuda()
    D = TinyP2PDiscriminator(in_channels=args.in_channels).cuda()
    wandb.watch((G, D))

    train_dataset, val_dataset = datasets.get_image_datasets(
        'cifar10',
        'none',
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    G_opt = Adam(G.parameters(), lr=0.0002)
    D_opt = Adam(D.parameters(), lr=1e-4)

    for epoch in range(args.num_epochs):
        print(f"starting epoch {epoch}/{args.num_epochs}")
        for i, (idx, img1, _, img2, _) in enumerate(tqdm(loader)):
            step_logs = {}
            # TODO: check if normalized
            img1 = (img1 * 2 - 1).cuda()
            img2 = (img2 * 2 - 1).cuda()
            # train generator
            G_opt.zero_grad()
            fake = G(img1)
            fake_scores = D(fake)

            # flipped labels, 0=real
            g_loss = F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores, device=fake_scores.device))
            g_loss.backward()
            G_opt.step()

            # train discriminator
            D_opt.zero_grad()
            fake = G(img2).detach()
            fake_scores = D(fake)
            real_scores = D(img2)

            # flipped labels, 0=real
            d_loss = 0.5 * (
                        F.binary_cross_entropy(fake_scores, torch.ones_like(fake_scores, device=fake_scores.device)) + \
                        F.binary_cross_entropy(real_scores, torch.zeros_like(real_scores, device=fake_scores.device)))
            d_loss.backward()
            D_opt.step()

            step_logs.update({"d_loss": d_loss, "g_loss": g_loss})
            if i % args.log_image_steps == 0:
                step_logs.update({"fake_train_images": wandb.Image(torchvision.utils.make_grid(fake[:64]))})
                step_logs.update({"real_train_images": wandb.Image(torchvision.utils.make_grid(img2[:64]))})
            if len(step_logs):
                wandb.log(step_logs)

        # saving, validation and metrics
        if (epoch+1) % args.log_fid_epochs == 0:
            fid = calculate_generator_fid(G, loader, "inception_cifar10_50000.pth", 50000)
            wandb.log({"fid":fid})
        if (epoch+1) % args.save_checkpoint_epochs == 0:
            os.makedirs(f"checkpoints/{wandb.run.name}", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'gen_state_dict': G.state_dict(),
                'gen_optimizer_state_dict': G_opt.state_dict(),
                'dis_state_dict': D.state_dict(),
                'dis_optimizer_state_dict': D_opt.state_dict(),
            }, f"checkpoints/{wandb.run.name}/state_{epoch:04d}.pth")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu', type=str)
    parser.add_argument(
        '--batch_size', type=int, default=128)
    parser.add_argument(
        '--num_workers', type=int, default=8)
    parser.add_argument(
        '--in_channels', type=int, default=3)
    parser.add_argument(
        '--num_epochs', type=int, default=500)
    parser.add_argument(
        '--log_image_steps', type=int, default=300)
    parser.add_argument(
        '--log_fid_epochs', type=int, default=10)
    parser.add_argument(
        '--save_checkpoint_epochs', type=int, default=100)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.manual_seed(20)
    wandb.init(name='AdamW', project="shahaf-TinyPixel2Pixel", entity="shafir")
    train_tiny_pix2pix(args)

