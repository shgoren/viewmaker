import os
from viewmaker.src.systems import image_systems,PretrainViewMakerSystemDABS
from viewmaker.src.utils.callbacks import MoCoLRScheduler
from viewmaker.src.utils.setup import process_config
import random, torch, numpy

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger,TensorBoardLogger
from dabs.src.datasets.catalog import MULTILABEL_DATASETS, PRETRAINING_DATASETS, UNLABELED_DATASETS


torch.backends.cudnn.benchmark = True

SYSTEM = {
    'PretrainViewMakerSystem': image_systems.PretrainViewMakerSystem,
    'PretrainViewMakerSystemDisc': image_systems.PretrainViewMakerSystemDisc,
    'PretrainNeuTraLADViewMakerSystem': image_systems.PretrainNeuTraLADViewMakerSystem,
    'PretrainExpertSystem': image_systems.PretrainExpertSystem,
    'TransferViewMakerSystem': image_systems.TransferViewMakerSystem,
    'TransferExpertSystem': image_systems.TransferExpertSystem,
    'PretrainExpertGANSystem': image_systems.PretrainExpertGANSystem,
    "PretrainViewMakerSystemFriendly": image_systems.PretrainViewMakerSystemFriendly,
    "PretrainViewMakerSystemDABS": PretrainViewMakerSystemDABS,
}


def run(args):
    '''Run the Lightning system. 

    Args:
        args
            args.config_path: str, filepath to the config file
        gpu_device: str or None, specifies GPU device as follows:
            None: CPU (specified as null in config)
            'cpu': CPU
            '-1': All available GPUs
            '0': GPU 0
            '4': GPU 4
            '0,3' GPUs 1 and 3
            See the following for more options: 
            https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html
    '''
    if args.gpu_device == 'cpu' or not args.gpu_device:
        gpu_device = None
    config = process_config(args.config, args)

    seed_everything(config.seed)
    SystemClass = SYSTEM[config.system]
    system = SystemClass(config)

    if config.optim_params.scheduler:
        lr_callback = globals()[config.optim_params.scheduler](
            initial_lr=config.optim_params.learning_rate,
            max_epochs=config.num_epochs,
            schedule=(
                int(0.6*config.num_epochs),
                int(0.8*config.num_epochs),
            ),
        )
        callbacks = [lr_callback]
    else:
        callbacks = []

    # TODO: adjust period for saving checkpoints.
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(config.exp_dir, 'checkpoints'),
        save_top_k=-1,
        every_n_epochs=config['copy_checkpoint_freq'],
    )
    callbacks.append(ckpt_callback)

    if not args.debug:
        wandblogger = WandbLogger(project=config.project, name=config.exp_name ,entity="shafir")
        wandblogger.log_hyperparams(config)
    else:
        wandblogger = TensorBoardLogger(config.exp_dir)

    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=len(args.gpu_device.split(",")),
         # 'ddp' is usually faster, but we use 'dp' so the negative samples 
         # for the whole batch are used for the SimCLR loss
        accelerator=config.distributed_backend or 'dp',
        max_epochs=config.num_epochs,
        min_epochs=config.num_epochs,
        checkpoint_callback=True,
        profiler="simple" if args.profiler else None,
        precision=config.optim_params.precision or 32,
        callbacks=callbacks,
        val_check_interval=config.val_check_interval or 1.0,
        limit_val_batches=config.limit_val_batches or 1.0,
        logger=wandblogger,
        log_every_n_steps=20
    )
    trainer.fit(system, ckpt_path=args.ckpt or config.continue_from_checkpoint)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='path to config file')
    parser.add_argument('--gpu-device', type=str, default=None)
    parser.add_argument('--profiler', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-t', type=float, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    args = parser.parse_args()

    # Ensure it's a string, even if from an older config
    gpu_device = str(args.gpu_device) if args.gpu_device else None
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    run(args)
