import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from BasicSR.basicsr.metrics.fid import load_patched_inception_v3, extract_inception_features
from viewmaker.src.datasets import datasets
from BasicSR.scripts.metrics.calculate_fid_stats_from_datasets import calculate_stats_from_dataset


def calculate_stats_from_dataset(data_loader, num_samples, dataset_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # inception model
    inception = load_patched_inception_v3(device)

    total_batch = math.ceil(num_samples / data_loader.batch_size)
    def data_generator():
        for idx, data in enumerate(data_loader):
            if idx >= total_batch:
                break
            else:
                yield data[1]

    features = extract_inception_features(
        data_generator(), inception, total_batch,
        device)
    features = features.numpy()
    total_len = features.shape[0]
    features = features[:num_samples]
    print(f'Extracted {total_len} features, '
          f'use the first {features.shape[0]} features to calculate stats.')
    mean = np.mean(features, 0)
    cov = np.cov(features, rowvar=False)

    save_path = f'inception_{dataset_name}_{num_samples}.pth'
    torch.save(
        dict(name=dataset_name, size=num_samples, mean=mean, cov=cov),
        save_path,
        _use_new_zipfile_serialization=False)


def calculate_cifar_fid_stats():
    train_dataset, val_dataset = datasets.get_image_datasets(
        'cifar10',
        'none',
    )

    loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=0,
    )
    calculate_stats_from_dataset(loader, 50000, "cifar10")


if __name__ == '__main__':
    calculate_cifar_fid_stats()
