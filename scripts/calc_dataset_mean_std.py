import torch
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import tqdm

if __name__ == '__main__':
    dataset = datasets.ImageFolder('/disk2/shahaf/FFHQ/',
                     transform=transforms.ToTensor())

    loader = data.DataLoader(dataset,
                             batch_size=10,
                             num_workers=0,
                             shuffle=False)
    mean = 0.
    std = 0.
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    print(mean)
    print(std)