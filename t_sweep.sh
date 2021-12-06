#!/bin/bash
python scripts/run_image.py config/image/pretrain_viewmaker_cifar10_neutralad.json --gpu-device 3 --num_workers 8 -t 0.01
python scripts/run_image.py config/image/pretrain_viewmaker_cifar10_neutralad.json --gpu-device 3 --num_workers 8 -t 0.05
python scripts/run_image.py config/image/pretrain_viewmaker_cifar10_neutralad.json --gpu-device 3 --num_workers 8 -t 0.1
python scripts/run_image.py config/image/pretrain_viewmaker_cifar10_neutralad.json --gpu-device 3 --num_workers 8 -t 0.3
python scripts/run_image.py config/image/pretrain_viewmaker_cifar10_neutralad.json --gpu-device 3 --num_workers 8 -t 0.5
python scripts/run_image.py config/image/pretrain_viewmaker_cifar10_neutralad.json --gpu-device 3 --num_workers 8 -t 0.7
python scripts/run_image.py config/image/pretrain_viewmaker_cifar10_neutralad.json --gpu-device 3 --num_workers 8 -t 1
python scripts/run_image.py config/image/pretrain_viewmaker_cifar10_neutralad.json --gpu-device 3 --num_workers 8 -t 2
python scripts/run_image.py config/image/pretrain_viewmaker_cifar10_neutralad.json --gpu-device 3 --num_workers 8 -t 5
python scripts/run_image.py config/image/pretrain_viewmaker_cifar10_neutralad.json --gpu-device 3 --num_workers 8 -t 10
python scripts/run_image.py config/image/pretrain_viewmaker_cifar10_neutralad.json --gpu-device 3 --num_workers 8 -t 30
python scripts/run_image.py config/image/pretrain_viewmaker_cifar10_neutralad.json --gpu-device 3 --num_workers 8 -t 100
