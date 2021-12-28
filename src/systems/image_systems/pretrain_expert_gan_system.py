from collections import OrderedDict

import numpy as np
import torch
from dotmap import DotMap

from ..image_systems import PretrainViewMakerSystem
from viewmaker.src.datasets import datasets
from viewmaker.src.objectives.infonce import NoiseConstrastiveEstimation
from viewmaker.src.objectives.memory_bank import MemoryBank
from viewmaker.src.objectives.simclr import SimCLRObjective
from viewmaker.src.utils import utils
from ...gans.tiny_pix2pix import TinyP2PGenerator


class PretrainExpertGANSystem(PretrainViewMakerSystem):
    '''Pytorch Lightning System for self-supervised pretraining
    with expert image views as described in Instance Discrimination
    or SimCLR.
    '''

    def __init__(self, config):
        super(PretrainViewMakerSystem, self).__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        self.loss_name = self.config.loss_params.name
        self.t = self.config.loss_params.t

        # DotMap is the default argument when a config argument is missing
        self.train_dataset, self.val_dataset = datasets.get_image_datasets(
            config.data_params.dataset,
            default_augmentations="gan_augmnetation",
        )
        train_labels = self.train_dataset.dataset.targets
        self.train_ordered_labels = np.array(train_labels)
        self.model = self.create_encoder()
        self.memory_bank = MemoryBank(
            len(self.train_dataset),
            self.config.model_params.out_dim,
        )
        self.G = self.load_pretrained_gan_augmenter()

    def forward(self, img):
        return self.model(img)

    def get_losses_for_batch(self, emb_dict, train=True):
        if self.loss_name == 'nce':
            loss_fn = NoiseConstrastiveEstimation(emb_dict['indices'], emb_dict['img_embs_1'], self.memory_bank,
                                                  k=self.config.loss_params.k,
                                                  t=self.t,
                                                  m=self.config.loss_params.m)
            loss = loss_fn.get_loss()
        elif self.loss_name == 'simclr':
            if 'img_embs_2' not in emb_dict:
                raise ValueError(f'img_embs_2 is required for SimCLR loss')
            loss_fn = SimCLRObjective(emb_dict['img_embs_1'], emb_dict['img_embs_2'], t=self.t)
            loss, acc = loss_fn.get_loss_and_acc()
        else:
            raise Exception(f'Objective {self.loss_name} is not supported.')

        if train:
            with torch.no_grad():
                if self.loss_name == 'nce':
                    new_data_memory = loss_fn.updated_new_data_memory()
                    self.memory_bank.update(emb_dict['indices'], new_data_memory)
                elif 'simclr' in self.loss_name:
                    outputs_avg = (utils.l2_normalize(emb_dict['img_embs_1'], dim=1) +
                                   utils.l2_normalize(emb_dict['img_embs_2'], dim=1)) / 2.
                    indices = emb_dict['indices']
                    self.memory_bank.update(indices, outputs_avg)
                else:
                    raise Exception(f'Objective {self.loss_name} is not supported.')

        return loss, acc

    def configure_optimizers(self):
        encoder_params = self.model.parameters()

        if self.config.optim_params.adam:
            optim = torch.optim.AdamW(encoder_params)
        else:
            optim = torch.optim.SGD(
                encoder_params,
                lr=self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
        return [optim], []

    def training_step(self, batch, batch_idx):
        emb_dict = {}
        indices, img, img2, neg_img, labels, = batch
        img2 = self.gan_augmnet(img2)
        if self.loss_name == 'nce':
            emb_dict['img_embs_1'] = self.forward(img)
        elif 'simclr' in self.loss_name:
            emb_dict['img_embs_1'] = self.forward(img)
            emb_dict['img_embs_2'] = self.forward(img2)

        emb_dict['indices'] = indices
        emb_dict['labels'] = labels
        return emb_dict

    def gan_augmnet(self, img):
        img = (img*2-1)
        img_aug = self.G(img)
        img_aug = ((img_aug+1)/2)
        return self.normalize(img_aug)

    def load_pretrained_gan_augmenter(self):
        G = TinyP2PGenerator(learn_residual=False, num_channels=3).cuda()
        G.load_state_dict(
            torch.load(
                "/disk2/shahaf/Apple/viewmaker/src/gans/checkpoints/original_implementaion/state_0479.pth"
            )["gen_state_dict"])
        return G

    def training_step_end(self, emb_dict):
        loss, acc = self.get_losses_for_batch(emb_dict, train=True)
        metrics = {'loss': loss, 'temperature': self.t, 'train_acc': acc}
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        emb_dict = {}
        indices, img, img2, neg_img, labels, = batch
        img2 = self.gan_augmnet(img2)
        img = self.normalize(img)
        if self.loss_name == 'nce':
            emb_dict['img_embs_1'] = self.forward(img)
        elif 'simclr' in self.loss_name:
            emb_dict['img_embs_1'] = self.forward(img)
            emb_dict['img_embs_2'] = self.forward(img2)

        emb_dict['indices'] = indices
        emb_dict['labels'] = labels
        img_embs = emb_dict['img_embs_1']

        loss, acc = self.get_losses_for_batch(emb_dict, train=False)

        num_correct, batch_size = self.get_nearest_neighbor_label(img_embs, labels)
        output = OrderedDict({
            'val_loss': loss,
            'val_encoder_loss': loss,
            'val_zero_knn_correct': torch.tensor(num_correct, dtype=float, device=self.device),
            'val_num_total': torch.tensor(batch_size, dtype=float, device=self.device),
            'val_encoder_acc': acc,
        })
        return output
