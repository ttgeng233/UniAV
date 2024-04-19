# python imports
import argparse
import os
import time
import datetime
from pprint import pprint
import logging
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

# our code
from libs.core import load_config
from libs.modeling import make_multimodal_meta_arch

from libs.utils import (LoadDatasetsTrain, LoadDatasetsVal, 
                        train_one_epoch_multitask, valid_one_epoch_multitask,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)
import libs.utils.task_utils as utils

def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    args.start_epoch = 0
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

    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
            device, n_gpu, bool(args.local_rank != -1)))

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    # fix the random seeds (this will fix everything)
    if default_gpu:
        logger.info('fix the random seeds...')
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    """2. create dataset / dataloader"""
    task_batch_size, task_num_iters, task_ids, task_datasets_train, task_dataloader_train = LoadDatasetsTrain(
        args, cfg, task_cfg, args.tasks.split("-"), rng_generator)
    
    if cfg['train_cfg']['evaluate']:
        task_datasets_val, task_dataloader_val, task_det_eval= LoadDatasetsVal(
            args, cfg, task_cfg, args.tasks.split("-"), split='val_split')

    task_names = []
    task_lr = []
    task_ave_iter = {}
    task_stop_controller = {}
    for task_id, num_iter in task_num_iters.items():
        task_names.append(task_cfg[task_id]['dataset_name'])
        task_lr.append(task_cfg[task_id]['learning_rate'])
        task_ave_iter[task_id] = int(
            task_cfg[task_id]['epochs'] * num_iter
            / args.num_train_epochs
            )
        task_stop_controller[task_id] = utils.MultiTaskStopOnPlateau(
            mode='max',
            patience=1,
            continue_threshold=0.005,  
            cooldown=1,
            threshold=0.001,)  
    
    # load text embeddings pre-extracted from ONE-PEACE text encoder for each dataset
    class_feature_anet = np.load('./data/activitynet13/anet_prompt.npy') 
    class_feature_unav = np.load('./data/unav100/unav100_prompt.npy')
    class_feature_dcase = np.load('./data/dcase/dcase_prompt.npy')
    task_cfg['TASK1']['clip_class_feature'] = torch.tensor(class_feature_anet, dtype=torch.float32).to(device)
    task_cfg['TASK2']['clip_class_feature'] = torch.tensor(class_feature_unav, dtype=torch.float32).to(device)
    task_cfg['TASK3']['clip_class_feature'] = torch.tensor(class_feature_dcase, dtype=torch.float32).to(device)
    
    logdir = os.path.join(ckpt_folder, 'logs')
    savePath = ckpt_folder
    tbLogger = utils.tbLogger(
            logdir,
            savePath,
            task_names,
            task_ids,
            task_num_iters,
            )

    """3. create model, optimizer, and scheduler"""
    if cfg['multi_modal']:
        model = make_multimodal_meta_arch(cfg['model_name'], **cfg['model'])
    
    model.to(device) 
    if args.local_rank != -1:
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(model, device_ids=[args.local_rank],
                                find_unused_parameters=True)
    elif n_gpu > 1:
        model = nn.DataParallel(model, device_ids=[args.local_rank])

    base_lr = min(task_lr)
    loss_scale = {}
    for i, task in enumerate(task_ids):
        loss_scale[task] = task_lr[i] / base_lr

    median_num_iter = sorted(task_ave_iter.values())[-1]

    # optimizer
    optimizer = make_optimizer(model, cfg['opt'], base_lr)
    # schedule
    scheduler = make_scheduler(optimizer, cfg['opt'], median_num_iter, args.num_train_epochs)

    # enable model EMA
    if default_gpu:
        logger.info("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    if default_gpu:
        with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
            pprint(cfg, stream=fid)
            fid.flush()

        """4. training / validation loop"""
        logger.info("\nStart training model {:s} ...".format(cfg['model_name']))


    max_epochs = args.num_train_epochs + cfg['opt']['warmup_epochs']
    best_mAP = {task_id: 0.0 for task_id in task_ids}

    for epoch in range(args.start_epoch, max_epochs):
        train_one_epoch_multitask(
            cfg,
            task_cfg,
            device,
            task_ids,
            median_num_iter,
            task_dataloader_train,
            model,
            optimizer,
            scheduler,
            epoch,
            default_gpu,
            loss_scale,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_logger=tbLogger,
            task_stop_controller=task_stop_controller,
            print_freq=args.print_freq
        )

        #evaluate on val set    
        if (epoch + 1) % cfg['train_cfg']['eval_freq'] == 0 or epoch == max_epochs - 1:
            if cfg['train_cfg']['evaluate']:
                if default_gpu:
                    logger.info("\nStart evaluating model {:s} ...".format(cfg['model_name']))
                start = time.time()
                avg_mAP = valid_one_epoch_multitask(
                    task_dataloader_val, 
                    task_cfg,
                    task_ids,
                    model, epoch, 
                    default_gpu,
                    task_stop_controller,
                    evaluator=task_det_eval, 
                    tb_writer=tbLogger, 
                    print_freq=args.print_freq
                    )
                end = time.time()
                if default_gpu:
                    logger.info("evluation done! Total time: {:0.2f} sec".format(end - start))
                
                if default_gpu:
                    for task_id in task_ids:
                        if avg_mAP[task_id] > best_mAP[task_id]:
                            best_mAP[task_id] = avg_mAP[task_id]
                            save_states = {
                                'epoch': epoch,
                                'state_dict': model.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'optimizer': optimizer.state_dict(),
                            }
                        save_states['state_dict_ema'] = model_ema.module.state_dict()
                
        if default_gpu:
            # save ckpt once in a while
            if (
                (epoch == max_epochs - 1) or
                (
                    (args.ckpt_freq > 0) and
                    (epoch % args.ckpt_freq == 0) and
                    (epoch > 0)
                )
            ):
                save_states = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                save_states['state_dict_ema'] = model_ema.module.state_dict()
                save_checkpoint(
                    save_states, None,
                    False,
                    file_folder=ckpt_folder,
                    file_name='epoch_{:03d}.pth.tar'.format(epoch)
                )    

    # wrap up
    tbLogger.txt_close()
    logger.info("All done!")
    return

if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        help='print frequency (default: 20 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--tasks', default='1', type=str,
                        help='task id list')  
    parser.add_argument('--num_train_epochs', default=40, type=int,
                        help='total number of training epochs')  
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='whether to use distributed training')
    args = parser.parse_args()
    main(args)
