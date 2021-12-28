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
from viewmaker.src.gans.calc_generator_fid import calculate_generator_fid
from viewmaker.src.gans.tiny_pix2pix import TinyP2PGenerator, TinyP2PDiscriminator


def train_tiny_pix2pix(args):
    G = TinyP2PGenerator(learn_residual=args.residual, num_channels=args.in_channels).cuda()
    torch.manual_seed(3)
    D = TinyP2PDiscriminator(in_channels=args.in_channels, wgan=args.wgan).cuda()
    if not args.debug:
        wandb.watch((G, D))
        wandb.config.update(args)

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

    G_opt = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_opt = Adam(D.parameters(), lr=1e-4)

    for epoch in range(args.num_epochs):
        print(f"starting epoch {epoch}/{args.num_epochs}")
        for i, (idx, img1, _, img2, _) in enumerate(tqdm(loader)):
            step_logs = {}
            # TODO: check if normalized
            # img1_n = (img1 * 2 - 1).cuda()
            # img2_n = (img2 * 2 - 1).cuda()
            img1_n = img1.cuda()
            img2_n = img2.cuda()
            # train generator
            G_opt.zero_grad()
            fake = G(img1_n)
            fake_scores = D(normalize(fake))
            if args.wgan:
                g_loss = -fake_scores.mean()
                g_acc = (fake_scores >= 0).float().mean()
            else:
                # flipped labels, real=0
                g_loss = F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores, device=fake_scores.device))
                g_acc = (fake_scores < 0.5).float().mean()
            g_loss.backward()
            G_opt.step()

            # train discriminator
            D_opt.zero_grad()
            fake = normalize(G(img2_n).detach())
            fake_scores = D(fake)
            if args.wgan:
                img2_n.requires_grad = True
            real_scores = D(normalize(img2_n))
            if args.wgan:
                disc_r1_penalty = D.r1_penalty(real_scores, img2_n)
            else:
                disc_r1_penalty = 0
            loss_n_acc = D.calc_loss_and_acc(real_scores, fake_scores,
                                             r1_penalty=disc_r1_penalty,
                                             r1_penalty_weight=args.r1_penalty_weight)

            loss_n_acc["d_total_loss"].backward()
            D_opt.step()

            step_logs.update(
                {"d_loss": loss_n_acc["d_loss"], "g_loss": g_loss, 'd_acc': loss_n_acc["d_acc"], 'g_acc': g_acc,
                 'disc_r1_penalty': loss_n_acc["disc_r1_penalty"], 'd_total_loss': loss_n_acc["d_total_loss"]})
            if i % args.log_image_steps == 0 and not args.debug:
                step_logs.update({"fake_train_images": wandb.Image(torchvision.utils.make_grid(fake[:64]))})
                step_logs.update({"real_train_images": wandb.Image(torchvision.utils.make_grid(img2_n[:64]))})
            if len(step_logs) and not args.debug:
                wandb.log(step_logs)

        # saving, validation and metrics
        if (epoch + 1) % args.log_fid_epochs == 0 and not args.debug:
            fid = calculate_generator_fid(G, loader, "inception_cifar10_50000.pth", 50000)
            wandb.log({"fid": fid})
        if (epoch + 1) % args.save_checkpoint_epochs == 0 and not args.debug:
            os.makedirs(f"checkpoints/{wandb.run.name}", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'gen_state_dict': G.state_dict(),
                'gen_optimizer_state_dict': G_opt.state_dict(),
                'dis_state_dict': D.state_dict(),
                'dis_optimizer_state_dict': D_opt.state_dict(),
            }, f"checkpoints/{wandb.run.name}/state_{epoch:04d}.pth")

def normalize(imgs):
    # These numbers were computed using compute_image_dset_stats.py
    mean = torch.tensor([0.491, 0.482, 0.446], device=imgs.device)
    std = torch.tensor([0.247, 0.243, 0.261], device=imgs.device)
    imgs = (imgs - mean[None, :, None, None]) / std[None, :, None, None]
    return imgs
    # return imgs * 2 - 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu', type=str)
    parser.add_argument(
        '--name', type=str, default=None)
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
        '--r1_penalty_weight', type=float, default=1)
    parser.add_argument(
        '--save_checkpoint_epochs', type=int, default=100)
    parser.add_argument(
        '--wgan', action='store_true')
    parser.add_argument(
        '--debug', action='store_true')
    parser.add_argument(
        '--residual', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.manual_seed(20)
    if not args.debug:
        wandb.init(name=args.name, project="shahaf-TinyPixel2Pixel", entity="shafir")
    train_tiny_pix2pix(args)
