import argparse
import os

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms
import torch.nn.functional as F
from sklearn.metrics import f1_score

from viewmaker.src.models.resnet import resnet50
import pytorch_lightning as pl
from pl_bolts.models.self_supervised.resnets import resnet18, resnet34, resnet50

class CIFAR100Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=100)

    def forward(self, x):
        return F.softmax(self.model.fc(self.model(x)[0]), dim=1)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        f1 = f1_score(y_hat.max(dim=1)[1].cpu(), y.cpu(), average="macro")
        self.log("train_loss", loss.detach())
        self.log("train_f1", f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        f1 = f1_score(y_hat.max(dim=1)[1].cpu(), y.cpu(), average="macro")
        self.log("val_loss", loss.detach())
        self.log("val_f1", f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=args.lr)


def cifar_100_loaders():
    cifar100_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                             std=(0.2675, 0.2565, 0.2761))
    ])
    data_path = "/disk3/shahaf/Apple/data/cifar100"
    train_ds = CIFAR100(root=data_path, train=True, download=True, transform=cifar100_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size)

    val_ds = CIFAR100(root=data_path, train=False, download=True, transform=cifar100_transforms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    return train_loader, val_loader


def main(args):
    # Init our model
    cifar100_model = CIFAR100Model()

    if not args.debug:
        wandblogger = WandbLogger(project='transfer_augmentations', name=args.exp_name)
        wandblogger.log_hyperparams(args)
    else:
        wandblogger = None

    # Init DataLoader from MNIST Dataset
    train_loader, test_loader = cifar_100_loaders()

    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.num_epochs,
        logger=wandblogger,
        progress_bar_refresh_rate=20,
    )

    # Train the model ⚡
    trainer.fit(cifar100_model, train_loader, test_loader)


if __name__ == "__main__":

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--exp_name", type=str)
    arg_parse.add_argument("--debug", action="store_true")
    arg_parse.add_argument("--num_epochs", type=int, default=50)
    arg_parse.add_argument("--batch_size", type=int, default=64)
    arg_parse.add_argument("--lr", type=float, default=0.01)
    arg_parse.add_argument("--gpu_device", type=str, default='0')
    args = arg_parse.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    main(args)
