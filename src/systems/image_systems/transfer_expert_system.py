import os

import torch
import torch.nn as nn
from dotmap import DotMap

from viewmaker.src.datasets import datasets
from viewmaker.src.utils import utils
from ..image_systems import TransferViewMakerSystem


class TransferExpertSystem(TransferViewMakerSystem):

    def __init__(self, config):
        super(TransferViewMakerSystem, self).__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size

        self.encoder, self.pretrain_config = self.load_pretrained_model()
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

        if not self.pretrain_config.model_params.resnet_small:
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # keep pooling layer

        # Freeze encoder for linear evaluation.
        self.encoder = self.encoder.eval()
        utils.frozen_params(self.encoder)

        default_augmentations = self.pretrain_config.data_params.default_augmentations
        if self.config.data_params.force_default_views or default_augmentations == DotMap():
            default_augmentations = 'all'
        self.train_dataset, self.val_dataset = datasets.get_image_datasets(
            config.data_params.dataset,
            default_augmentations=default_augmentations,
        )
        self.num_features = num_features
        self.model = self.create_model()

    def load_pretrained_model(self):
        base_dir = self.config.pretrain_model.exp_dir
        checkpoint_name = self.config.pretrain_model.checkpoint_name

        config_path = os.path.join(base_dir, 'config.json')
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
        return encoder, config

    def forward(self, img, unused_valid=None):
        del unused_valid
        batch_size = img.size(0)
        if self.pretrain_config.model_params.resnet_small:
            if self.config.model_params.use_prepool:
                embs = self.encoder(img, layer=5)
            else:
                embs = self.encoder(img, layer=6)
        else:
            embs = self.encoder(img)
        return self.model(embs.view(batch_size, -1))
