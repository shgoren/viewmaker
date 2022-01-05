from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
from dotmap import DotMap
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from torch import autograd
from torchvision import models
from torchvision.utils import make_grid

from viewmaker.src.datasets import datasets
from viewmaker.src.models import resnet_small
from viewmaker.src.models import viewmaker
from viewmaker.src.objectives.adversarial import AdversarialSimCLRLoss, AdversarialNCELoss
from viewmaker.src.objectives.memory_bank import MemoryBank
from viewmaker.src.utils import utils
from ..image_systems.utils import create_dataloader


class PretrainViewMakerSystem(pl.LightningModule):
    '''Pytorch Lightning System for self-supervised pretraining
    with adversarially generated views.
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        self.loss_name = self.config.loss_params.objective
        self.t = self.config.loss_params.t

        self.train_dataset, self.val_dataset = datasets.get_image_datasets(
            config.data_params.dataset,
            config.data_params.default_augmentations or 'none',
        )
        # Used for computing knn validation accuracy
        train_labels = self.train_dataset.dataset.targets
        self.train_ordered_labels = np.array(train_labels)

        self.budget_sched = config.model_params.budget_sched or False
        self.budget_steps = np.linspace(0.05, 1, 180)

        if config.model_params.pretrained or False:
            print("loading pretrained encoder")
            self.model = self.load_pretrained_encoder()
        else:
            self.model = self.create_encoder()
        self.viewmaker = self.create_viewmaker()

        # Used for computing knn validation accuracy.
        self.memory_bank = MemoryBank(
            len(self.train_dataset),
            self.config.model_params.out_dim,
        )


    def get_budget(self):
        if not self.budget_sched:
            return self.distortion_budget

        idx = self.current_epoch
        # before budget sched start
        if idx < 20:
            return self.budget_steps[0]
        if idx < len(self.budget_steps):
            return self.budget_steps[idx]
        else:
            return self.budget_steps[-1]

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

    def load_pretrained_encoder(self, freeze=True):
        resnet18 = models.resnet18(pretrained=True)
        return resnet18

    def create_viewmaker(self):
        if not isinstance(self.config.model_params.view_bound_magnitude, DotMap):
            self.config.model_params.additive_budget = self.config.model_params.view_bound_magnitude

        view_model = viewmaker.Viewmaker(
            num_channels=self.train_dataset.NUM_CHANNELS,
            activation=self.config.model_params.generator_activation or 'relu',
            clamp=self.config.model_params.clamp_views,
            frequency_domain=self.config.model_params.spectral or False,
            downsample_to=self.config.model_params.viewmaker_downsample or False,
            num_res_blocks=self.config.model_params.num_res_blocks or 5,
            use_budget=True,
            image_dim = (32,32),
            multiplicative=self.config.model_params.multiplicative or 0,
            multiplicative_budget=self.config.model_params.multiplicative_budget or 0.25,
            additive=self.config.model_params.additive or 1,
            additive_budget=self.config.model_params.additive_budget or 0.05,
            tps=self.config.model_params.tps or 0,
            tps_budget=self.config.model_params.tps_budget or 0.1,
            aug_proba=self.config.model_params.aug_proba or 1,
            budget_aware=self.config.model_params.budget_aware or False
        )
        return view_model

    def view(self, imgs, with_unnormalized = False):
        if 'Expert' in self.config.system:
            raise RuntimeError('Cannot call self.view() with Expert system')
        unnormalized = self.viewmaker(imgs)
        views = self.normalize(unnormalized)
        if with_unnormalized:
            return views, unnormalized
        return views

    def noise(self, batch_size, device):
        shape = (batch_size, self.config.model_params.noise_dim)
        # Center noise at 0 then project to unit sphere.
        noise = utils.l2_normalize(torch.rand(shape, device=device) - 0.5)
        return noise

    def get_repr(self, img):
        '''Get the representation for a given image.'''
        if 'Expert' not in self.config.system:
            # The Expert system datasets are normalized already.
            img = self.normalize(img)
        return self.model(img)

    def normalize(self, imgs):
        # These numbers were computed using compute_image_dset_stats.py
        if 'cifar' in self.config.data_params.dataset:
            mean = torch.tensor([0.491, 0.482, 0.446], device=imgs.device)
            std = torch.tensor([0.247, 0.243, 0.261], device=imgs.device)
        else:
            raise ValueError(f'Dataset normalizer for {self.config.data_params.dataset} not implemented')
        imgs = (imgs - mean[None, :, None, None]) / std[None, :, None, None]
        return imgs

    def forward(self, batch, train=True):
        indices, img, img2, neg_img, _, = batch
        if self.loss_name == 'AdversarialNCELoss':
            view1 = self.view(img)
            emb_dict = {
                'indices': indices,
                'view1_embs': self.model(view1),
                'orig_embs': self.get_repr(img)
            }

        elif self.loss_name == 'AdversarialSimCLRLoss':
            if self.config.model_params.double_viewmaker:
                view1, view2 = self.view(img)
            else:
                # return unnormalized for plotting
                view1,unnormalized_view1 = self.view(img,True)
                view2, unnormalized_view2 = self.view(img2, True)
            emb_dict = {
                'indices': indices,
                'view1_embs': self.model(view1),
                'view2_embs': self.model(view2),
                'orig_embs': self.get_repr(img)
            }
        else:
            raise ValueError(f'Unimplemented loss_name {self.loss_name}.')

        if self.global_step % 200 == 0:
            # Log some example views.
            # row of images, row of diff, row of view
            view1_diff = 1 - (unnormalized_view1[:10] - img[:10])
            view2_diff = 1 - (unnormalized_view2[:10] - img[:10])
            grid = make_grid(torch.cat([img[:10],
                                        unnormalized_view1[:10]/(img[:10]+1e-6),
                                        unnormalized_view1[:10],
                                        view1_diff]
                                       ),nrow=10)
            grid = torch.clamp(torchvision.transforms.Resize(512)(grid), 0, 1)
            view1_view2_grid = make_grid(torch.cat([view1_diff, view2_diff]), nrow=10)
            view1_view2_grid = torch.clamp(torchvision.transforms.Resize(512)(view1_view2_grid), 0, 1)
            if isinstance(self.logger, WandbLogger):
                wandb.log({"original_vs_views": wandb.Image(grid,
                                                            caption=f"Epoch: {self.current_epoch}, Step {self.global_step}"),
                           # "budget": self.viewmaker.distortion_budget,
                           "mean distortion": (1 - view1_diff).abs().mean(),
                           "view1_vs_view2": wandb.Image(view1_view2_grid,
                                                         caption=f"Epoch: {self.current_epoch}, Step {self.global_step}")
                           })
        return emb_dict

    def get_losses_for_batch(self, emb_dict, train=True):
        if self.loss_name == 'AdversarialSimCLRLoss':
            view_maker_loss_weight = self.config.loss_params.view_maker_loss_weight
            loss_function = AdversarialSimCLRLoss(
                embs1=emb_dict['view1_embs'],
                embs2=emb_dict['view2_embs'],
                t=self.t,
                view_maker_loss_weight=view_maker_loss_weight
            )
            encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = loss_function.get_loss()
            img_embs = emb_dict['orig_embs']
        elif self.loss_name == 'AdversarialNCELoss':
            view_maker_loss_weight = self.config.loss_params.view_maker_loss_weight
            loss_function = AdversarialNCELoss(
                emb_dict['indices'],
                emb_dict['view1_embs'],
                self.memory_bank,
                k=self.config.loss_params.k,
                t=self.t,
                m=self.config.loss_params.m,
                view_maker_loss_weight=view_maker_loss_weight
            )
            encoder_loss, view_maker_loss = loss_function.get_loss()
            img_embs = emb_dict['orig_embs']
        else:
            raise Exception(f'Objective {self.loss_name} is not supported.')

        # Update memory bank.
        if train:
            with torch.no_grad():
                if self.loss_name == 'AdversarialNCELoss':
                    new_data_memory = loss_function.updated_new_data_memory()
                    self.memory_bank.update(emb_dict['indices'], new_data_memory)
                else:
                    new_data_memory = utils.l2_normalize(img_embs, dim=1)
                    self.memory_bank.update(emb_dict['indices'], new_data_memory)
        return encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim

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

    def training_step(self, batch, batch_idx, optimizer_idx):
        # with autograd.detect_anomaly():
        emb_dict = self.forward(batch)
        emb_dict['optimizer_idx'] = torch.tensor(optimizer_idx, device=self.device)
        return emb_dict

    def training_step_end(self, emb_dict):
        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = self.get_losses_for_batch(emb_dict, train=True)

        if self.budget_sched:
            self.viewmaker.distortion_budget = self.get_budget()

        # assuming they fixed taht in new versions
        # # Handle Tensor (dp) and int (ddp) cases
        # if emb_dict['optimizer_idx'].__class__ == int or emb_dict['optimizer_idx'].dim() == 0:
        optimizer_idx = emb_dict['optimizer_idx']
        # else:
        #     optimizer_idx = emb_dict['optimizer_idx'][0]

        if optimizer_idx == 0:
            metrics = {
                'encoder_loss': encoder_loss, "train_acc": encoder_acc,
                "positive_sim":positive_sim, "negative_sim": negative_sim
            }
            loss = encoder_loss
        elif optimizer_idx == 1:
            metrics = {
                'view_maker_loss': view_maker_loss,
            }
            loss = view_maker_loss

        self.log_dict(metrics)
        return loss

    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu=False,
            using_native_amp=False,
            using_lbfgs=False,
    ):
        # update viwmaker every step
        if optimizer_idx == 0:
            super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu,
                                   using_native_amp, using_lbfgs)
        freeze_after_epoch = self.config.optim_params.viewmaker_freeze_epoch and \
                    self.current_epoch > self.config.optim_params.viewmaker_freeze_epoch

        if optimizer_idx == 1:
            if freeze_after_epoch:
                # freeze viewmaker after a certain number of epochs
                # call the closure by itself to run `training_step` + `backward` without an optimizer step
                optimizer_closure()
            else:
                super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu,
                                       using_native_amp, using_lbfgs)

    def validation_step(self, batch, batch_idx):
        emb_dict = self.forward(batch, train=False)
        if 'img_embs' in emb_dict:
            img_embs = emb_dict['img_embs']
        else:
            _, img, _, _, _ = batch
            img_embs = self.get_repr(img)  # Need encoding of image without augmentations (only normalization).
        labels = batch[-1]
        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = self.get_losses_for_batch(
            emb_dict, train=False)
        num_correct, batch_size = self.get_nearest_neighbor_label(img_embs, labels)

        if batch_idx==0:
            self.train_linear_probe()
        probe_score = self.linear_probe.score(img_embs.detach().cpu().numpy(), labels.cpu().numpy())
        probe_score = torch.Tensor([probe_score])

        output = OrderedDict({
            'val_encoder_loss': encoder_loss,
            'val_view_maker_loss': view_maker_loss,
            'val_zero_knn_correct': torch.tensor(num_correct, dtype=float, device=self.device),
            'val_num_total': torch.tensor(batch_size, dtype=float, device=self.device),
            'val_positive_sim': positive_sim,
            'val_negative_sim': negative_sim,
            'val_linear_probe_score': probe_score,
        })
        return output

    def train_linear_probe(self):
        from sklearn.linear_model import LogisticRegression
        train_X, train_y = self.memory_bank._bank.cpu().numpy(), self.train_ordered_labels
        self.linear_probe = LogisticRegression()
        self.linear_probe.fit(train_X, train_y)



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
        self.log_dict(metrics)
        return {'val_enc_loss': metrics['val_encoder_loss'],
                'progress_bar': progress_bar}

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

        enc_list = [encoder_optim, view_optim]

        return enc_list, []

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size,
                                 shuffle=False, drop_last=False)
