from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb

from viewmaker.src.datasets import datasets
from viewmaker.src.models import resnet_small
from viewmaker.src.models import viewmaker
from viewmaker.src.objectives.memory_bank import MemoryBank
from viewmaker.src.objectives.neutralad import NeuTraLADLoss
from viewmaker.src.objectives.simclr import SimCLRObjective
from viewmaker.src.utils import utils
from ..image_systems.utils import create_dataloader


class PretrainNeuTraLADViewMakerSystem(pl.LightningModule):
    '''Pytorch Lightning System for self-supervised pretraining
    with NeuTraL AD generated views.
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        self.loss_name = self.config.loss_params.objective
        if self.loss_name != 'NeuTraLADLoss':
            raise ValueError(f'This system is only for NeuTraL AD loss.')

        self.t = self.config.loss_params.t
        self.automatic_optimization = False  # This tells Lightning to let us control the training loop.
        torch.autograd.set_detect_anomaly(True)

        self.train_dataset, self.val_dataset = datasets.get_image_datasets(
            config.data_params.dataset,
            config.data_params.default_augmentations or 'none',
        )

        self.model = self.create_encoder()
        self.viewmaker = self.create_viewmaker()

        # Used for computing knn validation accuracy
        train_labels = self.train_dataset.dataset.targets
        self.train_ordered_labels = np.array(train_labels)
        self.memory_bank = MemoryBank(
            len(self.train_dataset),
            self.config.model_params.out_dim,
        )

    def view(self, imgs):
        if 'Expert' in self.config.system:
            raise RuntimeError('Cannot call self.view() with Expert system')
        views = self.viewmaker(imgs)
        return views

    def create_encoder(self):
        '''Create the encoder model.'''
        if self.config.model_params.resnet_small:
            # ResNet variant for smaller inputs (e.g. CIFAR-10).
            encoder_model = resnet_small.ResNet18(self.config.model_params.out_dim)
        else:
            resnet_class = getattr(
                torchvision.models,
                self.config.model_params.resnet_version,
            )
            encoder_model = resnet_class(
                pretrained=False,
                num_classes=self.config.model_params.out_dim,
            )
        if self.config.model_params.projection_head:
            mlp_dim = encoder_model.fc.weight.size(1)
            encoder_model.fc = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                encoder_model.fc,
            )
        return encoder_model

    def create_viewmaker(self):
        view_model = viewmaker.Viewmaker(
            num_channels=self.train_dataset.NUM_CHANNELS,
            distortion_budget=self.config.model_params.view_bound_magnitude,
            activation=self.config.model_params.generator_activation or 'relu',
            clamp=self.config.model_params.clamp_views,
            frequency_domain=self.config.model_params.spectral or False,
            downsample_to=self.config.model_params.viewmaker_downsample or False,
            num_res_blocks=self.config.model_params.num_res_blocks or 5,
            num_views=self.config.model_params.num_views or 2
        )
        return view_model

    def noise(self, batch_size, device):
        shape = (batch_size, self.config.model_params.noise_dim)
        # Center noise at 0 then project to unit sphere.
        noise = utils.l2_normalize(torch.rand(shape, device=device) - 0.5)
        return noise

    def normalize(self, imgs):
        # These numbers were computed using compute_image_dset_stats.py
        if 'cifar' in self.config.data_params.dataset:
            mean = torch.tensor([0.491, 0.482, 0.446], device=imgs.device)
            std = torch.tensor([0.247, 0.243, 0.261], device=imgs.device)
        else:
            raise ValueError(f'Dataset normalizer for {self.config.data_params.dataset} not implemented')
        imgs = (imgs - mean[None, :, None, None]) / std[None, :, None, None]
        return imgs

    def forward(self, img, view=True, log=True):
        if view:
            views = self.view(img)
        else:
            views = img
        embs = self.model(views)

        if self.global_step % 200 == 0 and self.model.training and log and view:
            log_image_cnt = (self.viewmaker.num_views+1)*3
            # Log some example views.
            # add a frame to mark the original images
            img = F.pad(img[:, :, 1:-1, 1:-1], (1, 1, 1, 1), "constant", value=0)
            # join images together
            views_to_log = torch.cat([views.view(-1, 2, *views.shape[1:]),
                                      img.unsqueeze(1)], dim=1) \
                .view(-1, *views.shape[1:])
            # fix dimensions and format
            views_to_log = views_to_log.permute(0, 2, 3, 1).detach().cpu().numpy()[:log_image_cnt]

            wandb.log({"examples": [wandb.Image(view_, caption=f"Epoch: {self.current_epoch}, Step {self.global_step}")
                                    for view_ in views_to_log]})
        return embs

    def get_enc_loss_and_acc(self, imgs):
        embs = self.forward(imgs, log=False)
        embs1, embs2 = embs.view(-1, 2, *embs.shape[1:]).permute(1, 0, 2)
        loss_function = SimCLRObjective(embs1, embs2, t=self.t)
        encoder_loss, encoder_acc = loss_function.get_loss_and_acc()
        return encoder_loss, encoder_acc

    def get_vm_loss(self, imgs):
        embs_orig = self.forward(imgs, view=False)
        embs_views = self.forward(imgs, view=True)
        embs1, embs2 = embs_views.view(-1, 2, *embs_views.shape[1:]).permute(1, 0, 2)
        loss_function = NeuTraLADLoss(embs_orig, embs1, embs2, t=self.t)
        vm_loss = loss_function.get_loss()
        return vm_loss, embs_orig

    def training_step(self, batch, batch_idx, optimizer_idx):
        indices, img, img2, neg_img, _, = batch
        encoder_loss, encoder_acc = self.get_enc_loss_and_acc(img)
        encoder_optim, vm_optim = self.trainer.optimizers
        encoder_loss.backward()

        # Alternate optimization steps between encoder and viewmaker.
        # Requires an extra forward + backward pass, but higher performance per step.
        encoder_optim.step()
        encoder_optim.zero_grad()
        vm_optim.zero_grad()

        # compute loss for the vm
        vm_loss, img_embs = self.get_vm_loss(img)
        vm_loss.backward()
        vm_optim.step()
        vm_optim.zero_grad()
        encoder_optim.zero_grad()

        self.memory_bank.update(indices, utils.l2_normalize(img_embs, dim=1))

        metrics = {
            'view_maker_loss': vm_loss,
            'encoder_loss': encoder_loss,
            'train_acc': encoder_acc,
            'temperature': self.t,
        }
        return {'loss': vm_loss, 'enc_loss': encoder_loss, 'log': metrics}

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        pass

    def validation_step(self, batch, batch_idx):
        indices, img, img2, neg_img, labels, = batch
        encoder_loss, encoder_acc = self.get_enc_loss_and_acc(img)
        view_maker_loss, img_embs = self.get_vm_loss(img)

        num_correct, batch_size = self.get_nearest_neighbor_label(img_embs, labels)

        output = OrderedDict({
            'val_loss': encoder_loss + view_maker_loss,
            'val_encoder_loss': encoder_loss,
            'val_view_maker_loss': view_maker_loss,
            'temperature': self.t,
            'val_encoder_acc': encoder_acc,
            'val_zero_knn_correct': torch.tensor(num_correct, dtype=float, device=self.device),
            'val_num_total': torch.tensor(batch_size, dtype=float, device=self.device),
        })
        return output

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            try:
                metrics[key] = torch.stack([elem[key] for elem in outputs]).mean()
            except:
                pass

        num_correct = torch.stack([out['val_zero_knn_correct'] for out in outputs]).sum()
        num_total = torch.stack([out['val_num_total'] for out in outputs]).sum()
        val_acc = num_correct / float(num_total)
        metrics['val_zero_knn_acc'] = val_acc
        progress_bar = {'acc': val_acc}
        return {'val_loss': metrics['val_loss'],
                'log': metrics,
                'progress_bar': progress_bar}

    def get_nearest_neighbor_label(self, img_embs, labels):
        '''
        Used for online kNN classifier.
        For each image in validation, find the nearest image in the
        training dataset using the memory bank. Assume its label as
        the predicted label.
        '''
        batch_size = img_embs.size(0)
        all_dps = self.memory_bank.get_all_dot_products(img_embs)
        _, neighbor_idxs = torch.topk(all_dps, k=1, sorted=False, dim=1)
        neighbor_idxs = neighbor_idxs.squeeze(1)
        neighbor_idxs = neighbor_idxs.cpu().numpy()

        neighbor_labels = self.train_ordered_labels[neighbor_idxs]
        neighbor_labels = torch.from_numpy(neighbor_labels).long()

        num_correct = torch.sum(neighbor_labels.cpu() == labels.cpu()).item()
        return num_correct, batch_size

    def configure_optimizers(self):
        # Optimize temperature with encoder.
        if type(self.t) == float or type(self.t) == int:
            encoder_params = self.model.parameters()
        else:
            encoder_params = list(self.model.parameters()) + [self.t]

        encoder_optim = torch.optim.SGD(
            encoder_params,
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum,
            weight_decay=self.config.optim_params.weight_decay,
        )
        view_optim_name = self.config.optim_params.viewmaker_optim
        view_parameters = self.viewmaker.parameters()
        if view_optim_name == 'adam':
            view_optim = torch.optim.Adam(
                view_parameters, lr=self.config.optim_params.viewmaker_learning_rate or 0.001)
        elif not view_optim_name or view_optim_name == 'sgd':
            view_optim = torch.optim.SGD(
                view_parameters,
                lr=self.config.optim_params.viewmaker_learning_rate or self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
        else:
            raise ValueError(f'Optimizer {view_optim_name} not implemented')

        return [encoder_optim, view_optim], []

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size,
                                 shuffle=False, drop_last=False)
