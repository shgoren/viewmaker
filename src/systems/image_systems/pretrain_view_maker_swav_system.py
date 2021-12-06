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

        self.model = self.create_encoder()
        self.viewmaker = self.create_viewmaker()
        if config.discriminator:
            self.disc = self.create_discriminator()
        else:
            self.disc = None
        self.disc_pause_steps = 0

        # Used for computing knn validation accuracy.
        self.memory_bank = MemoryBank(
            len(self.train_dataset),
            self.config.model_params.out_dim,
        )

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

    def create_discriminator(self):
        return resnet_small.CustomResnet(1, [1, 1, 1, 1])
        # return resnet_small.ResNet18(1)

    def create_viewmaker(self):
        view_model = viewmaker.Viewmaker(
            num_channels=self.train_dataset.NUM_CHANNELS,
            distortion_budget=self.config.model_params.view_bound_magnitude,
            activation=self.config.model_params.generator_activation or 'relu',
            clamp=self.config.model_params.clamp_views,
            frequency_domain=self.config.model_params.spectral or False,
            downsample_to=self.config.model_params.viewmaker_downsample or False,
            num_res_blocks=self.config.model_params.num_res_blocks or 5,
        )
        return view_model

    def view(self, imgs):
        if 'Expert' in self.config.system:
            raise RuntimeError('Cannot call self.view() with Expert system')
        views = self.viewmaker(imgs)
        views = self.normalize(views)
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
                view1 = self.view(img)
                view2 = self.view(img2)
            emb_dict = {
                'indices': indices,
                'view1_embs': self.model(view1),
                'view2_embs': self.model(view2),
                'orig_embs': self.get_repr(img)
            }
            if self.disc is not None:
                emb_dict["dis_real_score"] = F.sigmoid(self.disc(img))
                emb_dict["dis_fake_score"] = F.sigmoid(
                    torch.cat([self.disc(view1.detach()), self.disc(view2.detach())], dim=0))
                with torch.no_grad():
                    emb_dict["view_fake_score"] = F.sigmoid(torch.cat([self.disc(view1), self.disc(view2)], dim=0))
        else:
            raise ValueError(f'Unimplemented loss_name {self.loss_name}.')

        if self.global_step % 200 == 0:
            # Log some example views.
            views_to_log = view1.permute(0, 2, 3, 1).detach().cpu().numpy()[:10]
            wandb.log({"examples": [
                wandb.Image(view, caption=f"Epoch: {self.current_epoch}, Step {self.global_step}, Train {train}") for
                view in views_to_log]})

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
            encoder_loss, encoder_acc, view_maker_loss = loss_function.get_loss()
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

        if self.disc is None:
            disc_loss = 0
            disc_acc = 0
            view_adv_acc = 0
        else:
            dis_real_score = emb_dict['dis_real_score']
            dis_fake_score = emb_dict['dis_fake_score']
            view_fake_score = emb_dict['view_fake_score']
            disc_loss = F.binary_cross_entropy(dis_real_score, torch.ones_like(dis_real_score)) + \
                        F.binary_cross_entropy(dis_fake_score, torch.zeros_like(dis_fake_score))
            disc_acc = torch.cat((dis_real_score > 0.5, dis_fake_score <= 0.5), dim=0).float().mean()
            view_fake_loss = F.binary_cross_entropy(view_fake_score, torch.ones_like(view_fake_score))
            view_adv_acc = (dis_fake_score >= 0.5).float().mean()
            view_maker_loss = view_maker_loss + view_fake_loss * 6

        # Update memory bank.
        if train:
            with torch.no_grad():
                if self.loss_name == 'AdversarialNCELoss':
                    new_data_memory = loss_function.updated_new_data_memory()
                    self.memory_bank.update(emb_dict['indices'], new_data_memory)
                else:
                    new_data_memory = utils.l2_normalize(img_embs, dim=1)
                    self.memory_bank.update(emb_dict['indices'], new_data_memory)

        return encoder_loss, encoder_acc, view_maker_loss, view_adv_acc, disc_loss, disc_acc

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
        emb_dict = self.forward(batch)
        emb_dict['optimizer_idx'] = torch.tensor(optimizer_idx, device=self.device)
        return emb_dict

    def training_step_end(self, emb_dict):
        encoder_loss, encoder_acc, view_maker_loss, view_adv_acc, disc_loss, disc_acc = self.get_losses_for_batch(
            emb_dict, train=True)

        # Handle Tensor (dp) and int (ddp) cases
        if emb_dict['optimizer_idx'].__class__ == int or emb_dict['optimizer_idx'].dim() == 0:
            optimizer_idx = emb_dict['optimizer_idx']
        else:
            optimizer_idx = emb_dict['optimizer_idx'][0]
        if optimizer_idx == 0:
            metrics = {
                'encoder_loss': encoder_loss, 'temperature': self.t, "train_acc": encoder_acc
            }
            return {'loss': encoder_loss, 'log': metrics}
        elif optimizer_idx == 1:
            metrics = {
                'view_maker_loss': view_maker_loss,
                'view_maker_adv_acc': view_adv_acc
            }
            return {'loss': view_maker_loss, 'log': metrics}
        else:
            metrics = {
                'disc_acc': disc_acc,
                'disc_loss': disc_loss,
            }
            if view_adv_acc < 0.4:
                if self.disc_pause_steps == 0:
                    self.disc_pause_steps = 4
                    ret = {'loss': disc_loss, 'log': metrics}
                    return ret
                self.disc_pause_steps -= 1
            ret = {'loss': torch.tensor(0.0, device=disc_loss.device, requires_grad=True), 'log': metrics}
            return ret




    def validation_step(self, batch, batch_idx):
        emb_dict = self.forward(batch, train=False)
        if 'img_embs' in emb_dict:
            img_embs = emb_dict['img_embs']
        else:
            _, img, _, _, _ = batch
            img_embs = self.get_repr(img)  # Need encoding of image without augmentations (only normalization).
        labels = batch[-1]
        encoder_loss, encoder_acc, view_maker_loss, view_adv_acc, disc_loss, disc_acc = self.get_losses_for_batch(
            emb_dict, train=False)

        num_correct, batch_size = self.get_nearest_neighbor_label(img_embs, labels)
        output = OrderedDict({
            'val_loss': encoder_loss + view_maker_loss + disc_loss,
            'val_encoder_loss': encoder_loss,
            'val_disc_loss': disc_loss,
            'val_disc_acc': disc_acc,
            'val_view_maker_loss': view_maker_loss,
            'val_view_adv_acc': view_adv_acc,
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

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx,
                       second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        if not self.config.optim_params.viewmaker_freeze_epoch:
            super().optimizer_step(current_epoch, batch_nb, optimizer, optimizer_idx)
            return

        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()
        elif current_epoch < self.config.optim_params.viewmaker_freeze_epoch:
            # Optionally freeze the viewmaker at a certain pretraining epoch.
            optimizer.step()
            optimizer.zero_grad()

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

        if self.disc is not None:
            disc_optim = torch.optim.SGD(
                self.disc.parameters(),
                lr=self.config.optim_params.disc_learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
            enc_list.append(disc_optim)
        return enc_list, []

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size,
                                 shuffle=False, drop_last=False)
