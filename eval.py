# python imports
import argparse
import os
import glob
import time
from pprint import pprint
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

# our code
from libs.core import load_config
from libs.modeling import make_multimodal_meta_arch
from libs.utils import (fix_random_seed, LoadDatasetsVal, 
                        valid_one_epoch_multitask)

def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg, task_cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    if args.local_rank == -1: 
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    logger = logging.getLogger(f'LOG')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()   
    formatter = logging.Formatter(f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("device: {} n_gpu: {}, distributed testing: {}".format(
            device, n_gpu, bool(args.local_rank != -1)))
    
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    task_ids = []
    for task_id in args.tasks.split('-'):
        task = "TASK" + task_id
        task_ids.append(task)
        assert len(task_cfg[task]['test_split']) > 0, "Test set must be specified!"
    
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
        ckpt_file = ckpt_file_list[-1]

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    if default_gpu:
        logger.info('fix the random seeds...')
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    task_datasets_val, task_dataloader_val, task_det_eval= LoadDatasetsVal(
            args, cfg, task_cfg, args.tasks.split("-"), split='test_split')

    # load text embeddings pre-extracted from ONE-PEACE text encoder for each dataset
    class_feature_anet = np.load('./data/activitynet13/anet_prompt.npy')
    class_feature_unav = np.load('./data/unav100/unav100_prompt.npy')
    class_feature_dcase = np.load('./data/dcase/dcase_prompt.npy')
    task_cfg['TASK1']['clip_class_feature'] = torch.tensor(class_feature_anet, dtype=torch.float32).to(device)
    task_cfg['TASK2']['clip_class_feature'] = torch.tensor(class_feature_unav, dtype=torch.float32).to(device)
    task_cfg['TASK3']['clip_class_feature'] = torch.tensor(class_feature_dcase, dtype=torch.float32).to(device)

    """3. create model and evaluator"""
    if cfg['multi_modal']:
        model = make_multimodal_meta_arch(cfg['model_name'], **cfg['model'])

    model.to(device) 
    if args.local_rank != -1:
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(model, device_ids=[args.local_rank],
                                     find_unused_parameters=True)
    elif n_gpu > 1:
        model = nn.DataParallel(model, device_ids=[args.local_rank])

    """4. load ckpt"""
    if default_gpu:
        print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(device)
    )
    # load ema model instead
    if default_gpu:
        print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    """5. Test the model"""
    # if default_gpu:
    #     print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    avg_mAP = valid_one_epoch_multitask(
                task_dataloader_val, 
                task_cfg,
                task_ids,
                model, -1, 
                default_gpu=True,
                task_stop_controller = None,
                evaluator=task_det_eval, 
                tb_writer=None, 
                print_freq=args.print_freq
                )
    if default_gpu:
        end = time.time()
        print("All done! Total time: {:0.2f} sec".format(end - start))
    return

if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('--tasks', default='1', type=str,
                        help='task id list')  
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='whether to use distributed training')
    args = parser.parse_args()
    main(args)
