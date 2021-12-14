import copy
import json
import importlib
import os
os.chdir("/disk2/shahaf/Apple/stylegan2-ada-pytorch")
dnnlib = importlib.import_module("stylegan2-ada-pytorch.dnnlib")
misc = importlib.import_module("stylegan2-ada-pytorch.torch_utils.misc")
legacy = importlib.import_module("stylegan2-ada-pytorch.legacy")

def load_stylegan_disc(folder_path, resume_pkl,  gpu=0):
    resume_pkl = f"{folder_path}/{resume_pkl}"
    with open(f"{folder_path}/common_kwargs.json") as f:
        common_kwargs = json.load(f)
    with open(f"{folder_path}/D_kwargs.json") as f:
        D_kwargs = json.load(f)
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(gpu) # subclass of torch.nn.Module

    # Resume from existing pickle.
    print(f'Resuming from "{resume_pkl}"')
    with dnnlib.util.open_url(resume_pkl) as f:
        resume_data = legacy.load_network_pkl(f)
    misc.copy_params_and_buffers(resume_data['D'], D, require_all=False)
    return D

if __name__=='__main__':
    D = load_stylegan_disc("training-runs/00007-cifar10-cifar-batch32-noaug","network-snapshot-005200.pkl", 1)