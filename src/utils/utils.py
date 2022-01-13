import os
import json
import shutil
import torch
import numpy as np
from collections import Counter, OrderedDict
from dotmap import DotMap
from matplotlib import pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def copy_checkpoint(folder='./', filename='checkpoint.pth.tar',
                    copyname='copy.pth.tar'):
    shutil.copyfile(os.path.join(folder, filename),
                    os.path.join(folder, copyname))


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_json(f_path):
    with open(f_path, 'r') as f:
        return json.load(f)


def save_json(obj, f_path):
    with open(f_path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def frozen_params(module):
    for p in module.parameters():
        p.requires_grad = False


def free_params(module):
    for p in module.parameters():
        p.requires_grad = True


def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))


def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.item())
    return res

class delayed_linear_schedule:

    def __init__(self, start_val, stop_val, start_step, stop_step):
        self.start_step = start_step
        self.start_val = start_val
        self.stop_val = stop_val
        self.stop_step = stop_step
        self.ramp_up_vals = np.linspace(start_val, stop_val, stop_step-start_step)

    def __getitem__(self, step):
        if step < self.start_step:
            return self.start_val
        elif self.start_step <= step < self.stop_step:
            return self.ramp_up_vals[step-self.start_step]
        else:
            return self.stop_val


def load_viewmaker_from_checkpoint(viewmaker_cpkt, config_path, eval=True):
    # base_dir = "/".join(args.ckpt.split("/")[:-2])
    # config_path = os.path.join(base_dir, 'config.json')
    # with open(config_path, 'r') as f:
    #     config_json = json.load(f)
    # config = DotMap(config_json)
    # system = PretrainViewMakerSystem(config)
    # checkpoint = torch.load(viewmaker_cpkt, map_location="cuda:0")
    # system.load_state_dict(checkpoint['state_dict'], strict=False)
    # viewmaker = system.viewmaker.eval()
    # return viewmaker

    config_path =config_path
    with open(config_path, 'r') as f:
        config_json = json.load(f)
    config = DotMap(config_json)

    SystemClass = globals()[config.system]
    system = SystemClass(config)
    checkpoint = torch.load(viewmaker_cpkt, map_location="cuda:0")
    system.load_state_dict(checkpoint['state_dict'], strict=False)
    if eval:
        viewmaker = system.viewmaker.eval()
    return viewmaker