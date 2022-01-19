import torch
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import tqdm

from dabs.src import datasets

if __name__ == '__main__':
    data_calsses = [getattr(datasets, k) for k in datasets.__dict__.keys() if str.isupper(k[0])]
    for data_cls in data_calsses:
        print(f"**************  {data_cls.__name__} *************")
        try:
            dataset = data_cls("/disk2/ofirb/dabs/data", download=True)

            loader = data.DataLoader(dataset,
                                     batch_size=128,
                                     num_workers=16,
                                     shuffle=False)
            mean = 0.
            std = 0.
            for data_point in tqdm(loader):
                images = [d for d in data_point if len(d.shape)==4]
                assert len(images) == 1, "found more than one component in the data that fits image dimensions"
                images = images[0]
                batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
                images = images.view(batch_samples, images.size(1), -1)
                mean += images.mean(2).sum(0)
                std += images.std(2).sum(0)


            mean /= len(loader.dataset)
            std /= len(loader.dataset)

            print(mean)
            print(std)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise(e)
            else:
                print(str(e))
        finally:
            print("************************************************************\n\n")
