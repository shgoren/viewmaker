import argparse
import json
import time
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import wandb
from PIL import Image
import imutils
import matplotlib.image as mpimg
from collections import OrderedDict
from dotmap import DotMap
from skimage import io, transform
from math import *
import sys
import xml.etree.ElementTree as ET

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from viewmaker.src.systems.image_systems import PretrainViewMakerSystemDisc, PretrainViewMakerSystem
from glob import glob

from viewmaker.src.systems.image_systems.utils import heatmap_of_view_effect


def load_viewmaker_from_checkpoint(viewmaker_cpkt, config_path, eval=True):
    config_path = config_path
    with open(config_path, 'r') as f:
        config_json = json.load(f)
    config = DotMap(config_json)

    SystemClass = globals()[config.system]
    system = SystemClass(config)
    viewmaker = system.viewmaker
    checkpoint = torch.load(viewmaker_cpkt, map_location="cuda:0")
    d = dict([(k.replace("viewmaker.", ""), v) for k, v in checkpoint['state_dict'].items() if "viewmaker" in k])
    viewmaker.load_state_dict(d, strict=False)
    if eval:
        viewmaker = viewmaker.eval()
    return viewmaker

def main(args):
    img_paths = glob(args.imgdir+"/*")[:100]

    viewmaker = load_viewmaker_from_checkpoint(args.ckpt, args.config)
    viewmaker = viewmaker.cuda()
    amount_images = 10
    cur_views = []
    cur_images = []
    t = 0
    for imp in img_paths:
        t0 = time.time()
        img = cv2.imread(imp)[:, :, ::-1] / 255
        if args.img_size > 0:
            img = cv2.resize(img, (args.img_size, args.img_size))
        img = torch.Tensor(img).moveaxis(2,0).unsqueeze(0).cuda()
        cur_images.append(img.cpu())
        with torch.no_grad():
            cur_views.append(viewmaker(img).cpu())

        t1 = time.time()-t0
        t += t1
        if len(cur_views)==amount_images:
            t = t/10
            print("avg time", t)
            t=0
            cur_images = torch.cat(cur_images, 0)
            cur_views = torch.cat(cur_views, 0)
            grid = make_grid(torch.cat([cur_images, cur_views,
                                    heatmap_of_view_effect(cur_images, cur_views)],
                                   dim=0), nrow=amount_images)
            grid = torch.clamp(grid, 0, 1.0)
            grid = grid
            cur_views=[]


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--imgdir", type=str)
    arg_parse.add_argument("--gpu_device", type=str, default='1')
    arg_parse.add_argument("--ckpt", type=str, default=None)
    arg_parse.add_argument("--config", type=str, default=None)
    arg_parse.add_argument("--img_size", type=int, default=0)
    arg_parse.add_argument("--override_budget", type=float, default=0.0)
    args = arg_parse.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

    main(args)