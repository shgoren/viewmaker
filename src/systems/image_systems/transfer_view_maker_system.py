import os
import random
from collections import OrderedDict

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
from dotmap import DotMap
from sklearn.metrics import f1_score
from torchvision.utils import make_grid
import wandb

from viewmaker.src.datasets import datasets
from viewmaker.src.models.transfer import LogisticRegression
from viewmaker.src.utils import utils
from ..image_systems.utils import create_dataloader, heatmap_of_view_effect
from ..image_systems import PretrainViewMakerSystem, PretrainViewMakerSystemDisc


class TransferViewMakerSystem(pl.LightningModule):
    '''Pytorch Lightning System for linear evaluation of self-supervised
    pretraining with adversarially generated views.
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        self.encoder, self.viewmaker, self.system, self.pretrain_config = self.load_pretrained_model()
        resnet = self.pretrain_config.model_params.resnet_version

        if resnet == 'resnet18':
            if self.config.model_params.use_prepool:
                if self.pretrain_config.model_params.resnet_small:
                    num_features = 512 * 4 * 4
                else:
                    num_features = 512 * 7 * 7
            else:
                num_features = 512
        else:
            raise Exception(f'resnet {resnet} not supported.')
        self.train_dataset, self.val_dataset = datasets.get_image_datasets(
            config.data_params.dataset,
            default_augmentations=self.pretrain_config.data_params.default_augmentations or False,
        )
        if not self.pretrain_config.model_params.resnet_small:
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # keep pooling layer

        self.encoder = self.encoder.eval()
        self.viewmaker = self.viewmaker.eval()
        # linear evaluation freezes pretrained weights
        utils.frozen_params(self.encoder)
        utils.frozen_params(self.viewmaker)

        self.num_features = num_features
        self.model = self.create_model()

    def load_pretrained_model(self):
        base_dir = self.config.pretrain_model.exp_dir
        config_dir = self.config.pretrain_model.config_dir
        checkpoint_name = self.config.pretrain_model.checkpoint_name
        config_name = self.config.pretrain_model.config_name if self.config.pretrain_model.config_name else 'config.json'

        config_path = os.path.join(config_dir, config_name)
        config_json = utils.load_json(config_path)
        config = DotMap(config_json)

        if self.config.model_params.resnet_small:
            config.model_params.resnet_small = self.config.model_params.resnet_small

        SystemClass = globals()[config.system]
        system = SystemClass(config)
        checkpoint_file = os.path.join(base_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        system.load_state_dict(checkpoint['state_dict'], strict=False)

        encoder = system.model.eval()
        viewmaker = system.viewmaker.eval()

        return encoder, viewmaker, system, system.config

    def create_model(self):
        num_class = self.train_dataset.NUM_CLASSES
        model = LogisticRegression(self.num_features, num_class)
        return model

    def noise(self, batch_size):
        shape = (batch_size, self.pretrain_config.model_params.noise_dim)
        # Center noise at 0 then project to unit sphere.
        noise = utils.l2_normalize(torch.rand(shape) - 0.5)
        return noise

    def forward(self, img, valid=False):
        original_img = img
        batch_size = img.size(0)
        if self.pretrain_config.data_params.spectral_domain:
            img = self.system.normalize(img)
            img = dct.dct_2d(img)
            img = (img - img.mean()) / img.std()
        if not valid and not self.config.optim_params.no_views:
            img = self.viewmaker(img)
            if type(img) == tuple:
                idx = random.randint(0, 1)
                img = img[idx]

        ############################### log images ###########################
        logging_steps = 20
        if isinstance(self.logger, WandbLogger):
            logging_steps = 200

        if self.global_step % logging_steps == 0:
            amount_images = 10
            grid = make_grid(torch.cat([original_img[:amount_images], img[:amount_images],
                                        heatmap_of_view_effect(original_img[:amount_images], img[:amount_images])],
                                       dim=0), nrow=amount_images)
            grid = torch.clamp(grid, 0, 1.0)
            if isinstance(self.logger, WandbLogger):
                wandb.log(
                    {"db_examples": wandb.Image(grid, caption=f"Epoch: {self.current_epoch}, Step {self.global_step}")})
            else:
                self.logger.experiment.add_image('db_examples', grid, self.global_step)

        ############################### log images END ###########################

        if 'Expert' not in self.pretrain_config.system and not self.pretrain_config.data_params.spectral_domain:
            img = self.system.normalize(img)
        if self.pretrain_config.model_params.resnet_small:
            if self.config.model_params.use_prepool:
                embs = self.encoder(img, layer=5)
            else:
                embs = self.encoder(img, layer=6)
        else:
            embs = self.encoder(img)
        return self.model(embs.view(batch_size, -1))

    def get_losses_for_batch(self, batch, valid=False):
        _, img, _, _, label = batch
        logits = self.forward(img, valid)
        if self.train_dataset.MULTI_LABEL:
            return F.binary_cross_entropy(torch.sigmoid(logits).view(-1), label.view(-1).float())
        else:
            return F.cross_entropy(logits, label)

    def get_accuracies_for_batch(self, batch, valid=False):
        _, img, _, _, label = batch
        batch_size = img.size(0)
        logits = self.forward(img, valid)
        if self.train_dataset.MULTI_LABEL:
            preds = torch.round(torch.sigmoid(logits))
            preds = preds.long().cpu()
            num_correct = torch.sum(preds.cpu() == label.cpu(), dim=0)
            num_correct = num_correct.detach().cpu().numpy()
            num_total = batch_size
            return num_correct, num_total, preds, label.cpu()
        else:
            preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            preds = preds.long().cpu()
            num_correct = torch.sum(preds == label.long().cpu()).item()
            num_total = batch_size
            return num_correct, num_total

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        with torch.no_grad():
            if self.train_dataset.MULTI_LABEL:
                num_correct, num_total, _, _ = self.get_accuracies_for_batch(batch)
                num_correct = num_correct.mean()
            else:
                num_correct, num_total = self.get_accuracies_for_batch(batch)
            metrics = {
                'train_loss': loss.detach(),
                'train_num_correct': torch.tensor(num_correct, dtype=float, device=self.device).detach(),
                'train_num_total': torch.tensor(num_total, dtype=float, device=self.device).detach(),
                'train_acc': torch.tensor(num_correct / float(num_total), dtype=float, device=self.device).detach()
            }
        self.log_dict(metrics)
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, valid=True)
        if self.train_dataset.MULTI_LABEL:  # regardless if binary or not
            num_correct, num_total, val_preds, val_labels = \
                self.get_accuracies_for_batch(batch, valid=True)
            return OrderedDict({
                'val_loss': loss,
                'val_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
                'val_num_total': torch.tensor(num_total, dtype=float, device=self.device),
                'val_acc': torch.tensor(num_correct / float(num_total), dtype=float, device=self.device),
                'val_pred_labels': val_preds.float(),
                'val_true_labels': val_labels.float(),
            })
        else:
            num_correct, num_total = self.get_accuracies_for_batch(batch, valid=True)
            return OrderedDict({
                'val_loss': loss,
                'val_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
                'val_num_total': torch.tensor(num_total, dtype=float, device=self.device),
                'val_acc': torch.tensor(num_correct / float(num_total), dtype=float, device=self.device),
            })

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            try:
                metrics[key] = torch.tensor([elem[key] for elem in outputs]).float().mean()
            except:
                pass

        if self.train_dataset.MULTI_LABEL:
            num_correct = torch.stack([out['val_num_correct'] for out in outputs], dim=1).sum(1)
            num_total = torch.stack([out['val_num_total'] for out in outputs]).sum()
            val_acc = num_correct / float(num_total)
            metrics['val_acc'] = val_acc.mean()
            progress_bar = {'acc': val_acc.mean()}
            num_class = self.train_dataset.NUM_CLASSES
            for c in range(num_class):
                val_acc_c = num_correct[c] / float(num_total)
                metrics[f'val_acc_feat{c}'] = val_acc_c
            val_pred_labels = torch.cat([out['val_pred_labels'] for out in outputs], dim=0).numpy()
            val_true_labels = torch.cat([out['val_true_labels'] for out in outputs], dim=0).numpy()

            val_f1 = 0
            for c in range(num_class):
                val_f1_c = f1_score(val_true_labels[:, c], val_pred_labels[:, c])
                metrics[f'val_f1_feat{c}'] = val_f1_c
                val_f1 = val_f1 + val_f1_c
            val_f1 = val_f1 / float(num_class)
            metrics['val_f1'] = val_f1
            progress_bar['f1'] = val_f1
            self.log_dict(metrics)
            return {'val_loss': metrics['val_loss'],
                    'log': metrics,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'progress_bar': progress_bar}
        else:
            num_correct = sum([out['val_num_correct'] for out in outputs])
            num_total = sum([out['val_num_total'] for out in outputs])
            val_acc = num_correct / float(num_total)
            metrics['val_acc'] = val_acc
            progress_bar = {'acc': val_acc}
            self.log_dict(metrics)
            return {'val_loss': metrics['val_loss'],
                    'log': metrics,
                    'val_acc': val_acc,
                    'progress_bar': progress_bar}

    def configure_optimizers(self):
        params_iterator = self.model.parameters()
        if self.config.optim_params == 'adam':
            optim = torch.optim.Adam(params_iterator)
        else:
            optim = torch.optim.SGD(
                params_iterator,
                lr=self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
        return [optim], []

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size,
                                 shuffle=False, drop_last=False)
