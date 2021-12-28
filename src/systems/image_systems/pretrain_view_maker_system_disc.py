from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from dotmap import DotMap
from pytorch_lightning.loggers import WandbLogger
from torch import autograd
from torchvision.utils import make_grid

from viewmaker.src.datasets import datasets
from viewmaker.src.gans.tiny_pix2pix import TinyP2PDiscriminator
from viewmaker.src.models import resnet_small
from viewmaker.src.models import viewmaker
from viewmaker.src.objectives.adversarial import AdversarialSimCLRLoss, AdversarialNCELoss
from viewmaker.src.objectives.memory_bank import MemoryBank
from viewmaker.src.utils import utils
from viewmaker.src.systems.image_systems.utils import create_dataloader
from pl_bolts.models.self_supervised import SimCLR


class PretrainViewMakerSystemDisc(pl.LightningModule):
    '''Pytorch Lightning System for self-supervised pretraining
    with adversarially generated views.
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        self.loss_name = self.config.loss_params.objective
        self.t = self.config.loss_params.t
        self.adv_loss_weight = self.config.disc.vm_gen_loss_weight
        self.r1_penalty_weight = self.config.disc.r1_penalty_weight
        self.budget_mean, self.budget_std = self.config.model_params.budget_mean, self.config.model_params.budget_std

        self.train_dataset, self.val_dataset = datasets.get_image_datasets(
            config.data_params.dataset,
            config.data_params.default_augmentations or 'none',
        )
        # Used for computing knn validation accuracy
        train_labels = self.train_dataset.dataset.targets
        self.train_ordered_labels = np.array(train_labels)

        self.budget_sched = config.model_params.budget_sched or False

        if config.model_params.pretrained or False:
            self.model = self.load_pretrained_encoder()
        else:
            self.model = self.create_encoder()
        self.viewmaker = self.create_viewmaker()
        if config.discriminator or False:
            self.disc = self.create_discriminator()
        else:
            self.disc = None

        # Used for computing knn validation accuracy.
        self.memory_bank = MemoryBank(
            len(self.train_dataset),
            self.config.model_params.out_dim,
        )

    def get_budget(self):
        if self.budget_sched == 'random_normal':
            return max(np.random.normal(self.budget_mean, self.budget_std), 0)

        elif self.budget_sched == 'linear':
            if not hasattr(self, "budget_steps"): # TODO: untested
                self.budget_steps = np.linspace(0.05, 1, 180)
            if not self.budget_sched:
                return self.distortion_budget
            idx = self.current_epoch
            # before budget sched start
            if idx < 30:
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
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
        if freeze:
            simclr.freeze()
        return simclr

    def create_discriminator(self):
        torch.manual_seed(3)
        return TinyP2PDiscriminator(wgan=self.config.disc.wgan, blocks_num=self.config.disc.conv_blocks)

    def create_viewmaker(self):
        VMClass = viewmaker.VIEWMAKERS[self.config.model_params.viewmaker_backbone]
        view_model = VMClass(
            num_channels=self.train_dataset.NUM_CHANNELS,
            distortion_budget=self.config.model_params.view_bound_magnitude,
            activation=self.config.model_params.generator_activation or 'relu',
            clamp=self.config.model_params.clamp_views or True,
            frequency_domain=self.config.model_params.spectral or False,
            downsample_to=self.config.model_params.viewmaker_downsample or False,
            num_res_blocks=self.config.model_params.num_res_blocks or 5,
            use_budget=self.config.model_params.use_budget,
            budget_aware=self.config.model_params.budget_aware
        )
        return view_model

    def view(self, imgs):
        if 'Expert' in self.config.system:
            raise RuntimeError('Cannot call self.view() with Expert system')
        views_unn = self.viewmaker(imgs)
        views = self.normalize(views_unn)
        return views, views_unn

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
        # return imgs * 2 - 1

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
                view1, unnormalized_view1 = self.view(img)
                view2, unnormalized_view2 = self.view(img2)
            emb_dict = {
                'indices': indices,
                'view1_embs': self.model(view1),
                'view2_embs': self.model(view2),
                'orig_embs': self.get_repr(img)
            }
            if self.disc is not None:
                img.requires_grad = True
                emb_dict["real_score"] = self.disc(self.normalize(img))
                emb_dict["fake_score"] = torch.cat([self.disc(view1), self.disc(view2)], dim=0)
                emb_dict['disc_r1_penalty'] = 0
                if self.disc.wgan:
                    try:
                        emb_dict["disc_r1_penalty"] = self.disc.r1_penalty(emb_dict["real_score"], img)
                        # this fails if in validation mode
                    except RuntimeError as e:
                        pass

        else:
            raise ValueError(f'Unimplemented loss_name {self.loss_name}.')

        if self.global_step % 200 == 0:
            # Log some example views.
            # row of images, row of diff, row of view
            grid = make_grid(torch.cat([img[:10], unnormalized_view1[:10],
                                        1 - (unnormalized_view1[:10] - img[:10])]), nrow=10)
            grid = torch.clamp(torchvision.transforms.Resize(512)(grid), 0, 1)
            if isinstance(self.logger, WandbLogger):
                wandb.log({"original_vs_views": wandb.Image(grid,
                                                            caption=f"Epoch: {self.current_epoch}, Step {self.global_step}")})

            view1_view2_grid = make_grid(torch.cat([1 - (unnormalized_view1[:10] - img[:10]),
                                        1 - (unnormalized_view2[:10] - img[:10])]), nrow=10)
            view1_view2_grid = torch.clamp(torchvision.transforms.Resize(512)(view1_view2_grid), 0, 1)
            if isinstance(self.logger, WandbLogger):
                wandb.log({"view1_vs_view2": wandb.Image(view1_view2_grid,
                                                            caption=f"Epoch: {self.current_epoch}, Step {self.global_step}")})
            # else:
            #     self.logger.experiment.add_image('original_vs_views', grid, self.global_step)

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
            gen_loss = 0
            gen_acc = 0
            disc_r1_penalty = 0
        else:
            real_s = emb_dict['real_score']
            fake_s = emb_dict['fake_score']
            loss_n_acc = self.disc.calc_loss_and_acc(real_s, fake_s,
                                                     r1_penalty=emb_dict['disc_r1_penalty'],
                                                     r1_penalty_weight=self.r1_penalty_weight)

            disc_loss = loss_n_acc["d_loss"]
            disc_acc = loss_n_acc["d_acc"]
            gen_loss = loss_n_acc["g_loss"]
            gen_acc = loss_n_acc["g_acc"]
            disc_r1_penalty = emb_dict['disc_r1_penalty']

        # Update memory bank.
        if train:
            with torch.no_grad():
                if self.loss_name == 'AdversarialNCELoss':
                    new_data_memory = loss_function.updated_new_data_memory()
                    self.memory_bank.update(emb_dict['indices'], new_data_memory)
                else:
                    new_data_memory = utils.l2_normalize(img_embs, dim=1)
                    self.memory_bank.update(emb_dict['indices'], new_data_memory)

        losses = {'encoder_loss': encoder_loss, 'encoder_acc': encoder_acc, 'view_maker_loss': view_maker_loss,
                  'gen_loss': gen_loss, 'disc_loss': disc_loss, 'disc_acc': disc_acc, 'gan_acc': gen_acc,
                  'disc_r1_penalty': disc_r1_penalty}
        return losses

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
        losses = self.get_losses_for_batch(emb_dict, train=True)

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
                'encoder_loss': losses['encoder_loss'], "train_acc": losses['encoder_acc']
            }
            loss = losses['encoder_loss']

        elif optimizer_idx == 1:

            loss = self.get_vm_loss_weight() * losses['view_maker_loss'] + self.adv_loss_weight * losses['gen_loss']
            metrics = {
                'view_maker_loss': losses['view_maker_loss'],
                'generator_loss': losses['gen_loss'],
                'view_maker_total_loss': loss
            }

        elif optimizer_idx == 2:
            metrics = {
                'disc_acc': losses['disc_acc'],
                'disc_loss': losses['disc_loss'],
                'disc_r1_penalty': losses['disc_r1_penalty']}
            loss = losses['disc_loss'] + self.r1_penalty_weight * losses['disc_r1_penalty']

        self.log_dict(metrics)
        return loss

    def get_vm_loss_weight(self):
        if self.current_epoch < self.config.disc.gan_warmup:
            return 0
        else:
            return 1

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        # update viwmaker every step
        if optimizer_idx == 0:
            super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu,
                                   using_native_amp, using_lbfgs)
        freeze_after_epoch = isinstance(self.config.optim_params.viewmaker_freeze_epoch, int) and \
                             self.current_epoch > self.config.optim_params.viewmaker_freeze_epoch

        if optimizer_idx == 1:
            if freeze_after_epoch:
                # freeze viewmaker after a certain number of epochs
                # call the closure by itself to run `training_step` + `backward` without an optimizer step
                optimizer_closure()
            else:
                super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu,
                                       using_native_amp, using_lbfgs)

        if optimizer_idx == 2:
            if freeze_after_epoch:
                # freeze viewmaker after a certain number of epochs
                # call the closure by itself to run `training_step` + `backward` without an optimizer step
                optimizer_closure()
            elif batch_idx % (self.config.disc.dis_skip_steps + 1) == 0:
                super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu,
                                       using_native_amp, using_lbfgs)
            else:
                optimizer_closure()

    def validation_step(self, batch, batch_idx):
        emb_dict = self.forward(batch, train=False)
        if 'img_embs' in emb_dict:
            img_embs = emb_dict['img_embs']
        else:
            _, img, _, _, _ = batch
            img_embs = self.get_repr(img)  # Need encoding of image without augmentations (only normalization).
        labels = batch[-1]
        losses = self.get_losses_for_batch(emb_dict,
                                           train=False)

        num_correct, batch_size = self.get_nearest_neighbor_label(img_embs, labels)
        output = OrderedDict({
            'val_encoder_loss': losses['encoder_loss'],
            'val_view_maker_loss': losses['view_maker_loss'],
            'val_generator_loss': losses['gen_loss'],
            'val_disc_loss': losses['disc_loss'],
            'val_disc_acc': losses['disc_acc'],
            'val_encoder_acc': losses['encoder_acc'],
            'val_zero_knn_correct': torch.tensor(num_correct, dtype=float, device=self.device),
            'val_num_total': torch.tensor(batch_size, dtype=float, device=self.device)
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

        if self.disc is not None:
            disc_optim = torch.optim.Adam(self.disc.parameters(),
                                          lr=self.config.optim_params.disc_learning_rate)
            enc_list.append(disc_optim)
        return enc_list, []

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size, shuffle=False)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size,
                                 shuffle=False, drop_last=False)
