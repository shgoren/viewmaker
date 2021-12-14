import argparse
import math
import numpy as np
import torch

from BasicSR.basicsr.metrics.fid import load_patched_inception_v3, extract_inception_features, calculate_fid


def calculate_generator_fid(generator, data_loader, dataset_fid_stats, num_sample=50000):
    """
    :param generator: data generator
    :param inputs: inputs for the generator
    :param dataset_fid_stats: Path to the dataset fid statistics.
    :param num_sample: number of samples for the dataset statistics calculation
    :return:
    """

    # inception model
    inception = load_patched_inception_v3(0)

    total_batch = math.ceil(num_sample / data_loader.batch_size)

    def sample_generator():
        for i, input_batch in enumerate(data_loader):
            imgs = input_batch[1].cuda()
            if i * imgs.size(0) > num_sample:
                break
            with torch.no_grad():
                samples = generator(imgs)
            yield samples

    features = extract_inception_features(
        sample_generator(), inception, total_batch, 0)
    features = features.numpy()
    total_len = features.shape[0]
    features = features[:num_sample]
    print(f'Extracted {total_len} features, '
          f'use the first {features.shape[0]} features to calculate stats.')
    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    # load the dataset stats
    stats = torch.load(dataset_fid_stats)
    real_mean = stats['mean']
    real_cov = stats['cov']

    # calculate FID metric
    fid = calculate_fid(sample_mean, sample_cov, real_mean, real_cov)
    return fid

if __name__ == '__main__':
    calculate_fid()
