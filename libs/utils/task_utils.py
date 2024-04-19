import logging
from tensorboardX import SummaryWriter
from functools import partial, wraps
from torch._six import inf
import torch.distributed as dist
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from libs.datasets import make_dataset
from libs.utils import ANETdetection
from libs.datasets.data_utils import trivial_batch_collator, worker_init_reset_seed

logger = logging.getLogger(f'LOG') 
logger.propagate = False

def LoadDatasetsTrain(args, cfg, task_cfg, ids, generator):

    task_datasets_train = {}
    task_dataloader_train = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_ids.append(task)
        batch_size = task_cfg[task]['batch_size'] 
        num_workers = cfg['num_workers']
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())
            num_workers = int(num_workers / dist.get_world_size())
        task_datasets_train[task] =None
        task_datasets_train[task] = make_dataset(
                        task_cfg[task]['dataset_name'], 
                        True, 
                        task_cfg[task]['train_split'], 
                        **task_cfg[task]['dataset'])
        task_num_iters[task] = 0
        task_batch_size[task] = 0

        if args.local_rank != -1:
            train_sampler = DistributedSampler(task_datasets_train[task])
        else:
            train_sampler = RandomSampler(task_datasets_train[task])

        task_dataloader_train[task] = DataLoader(
                                        task_datasets_train[task],
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        sampler=train_sampler,   
                                        collate_fn=trivial_batch_collator,
                                        worker_init_fn=worker_init_reset_seed, 
                                        drop_last=True,
                                        generator=generator,
                                        persistent_workers=True
                                        )
        task_num_iters[task] = len(task_dataloader_train[task])
        task_batch_size[task] = batch_size
        logger.info("load dataset: task: {}  batch_size: {}  num_workers: {}".format(
            task, batch_size, num_workers))
    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_train,
        task_dataloader_train,
        )

def LoadDatasetsVal(args, cfg, task_cfg, ids, split):
    task_datasets_val = {}
    task_dataloader_val = {}
    task_ids = []
    task_val_db_vars ={}
    task_det_eval = {}

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_ids.append(task)
        batch_size = task_cfg[task]['batch_size'] 
        num_workers = cfg['num_workers']
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())
            num_workers = int(num_workers / dist.get_world_size())
        task_datasets_val[task] =None
        task_datasets_val[task] = make_dataset(
                        task_cfg[task]['dataset_name'], 
                        False, 
                        task_cfg[task][split], 
                        **task_cfg[task]['dataset'])
        task_dataloader_val[task] = DataLoader(
                                        task_datasets_val[task],
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        collate_fn=trivial_batch_collator,
                                        shuffle=False, 
                                        drop_last=False,
                                        persistent_workers=True
                                        )
        task_val_db_vars[task] = task_datasets_val[task].get_attributes()
        task_det_eval[task] = ANETdetection(
            task_datasets_val[task].json_file,
            task_datasets_val[task].split[0],
            tiou_thresholds = task_val_db_vars[task]['tiou_thresholds']
        )
    return (
        task_datasets_val,
        task_dataloader_val,
        task_det_eval
        )

def ForwardModelsTrain(
        cfg, task_cfg, device, task_id, task_count,
        task_iter_train, task_dataloader_train, model,
        ):
    # reset the task iteration when needed.
    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        task_iter_train[task_id] = iter(task_dataloader_train[task_id])

    task_count[task_id] += 1
    # get the batch
    batch = task_iter_train[task_id].next()

    losses = model(batch, task_id, task_cfg[task_id]['task_type'])

    return losses
    
class tbLogger(object):
    def __init__(
        self,
        log_dir,
        txt_dir,
        task_names,
        task_ids,
        task_num_iters,
        gradient_accumulation_steps=1,
        save_logger=True,
        txt_name="out.txt",
    ):
        logger.info("logging file at: " + log_dir)

        self.save_logger = save_logger
        self.log_dir = log_dir
        self.txt_dir = txt_dir
        if self.save_logger:
            self.logger = SummaryWriter(log_dir=log_dir)

        self.txt_f = open(txt_dir + "/" + txt_name, "w")
        self.task_id2name = {
            ids: name.replace("+", "plus") for ids, name in zip(task_ids, task_names)
        }
        self.task_ids = task_ids
        self.task_loss = {task_id: 0 for task_id in task_ids}
        self.task_loss_tmp = {task_id: 0 for task_id in task_ids}
        self.task_score_tmp = {task_id: 0 for task_id in task_ids}
        self.task_norm_tmp = {task_id: 0 for task_id in task_ids}

        self.task_lr = {task_id: 0 for task_id in task_ids}
        self.task_lr_tmp = {task_id: 0 for task_id in task_ids}

        self.task_step = {task_id: 0 for task_id in task_ids}
        self.task_step_tmp = {task_id: 0 for task_id in task_ids}
        self.task_num_iters = task_num_iters
        self.epochId = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.task_reg_loss_val = {task_id: 0 for task_id in task_ids}
        self.task_cls_loss_val = {task_id: 0 for task_id in task_ids}
        self.task_final_loss_val = {task_id: 0 for task_id in task_ids}
       
        self.task_loss_val = {task_id: 0 for task_id in task_ids}
        self.task_score_val = {task_id: 0 for task_id in task_ids}
        self.task_step_val = {task_id: 0 for task_id in task_ids}
        self.task_iter_val = {task_id: 0 for task_id in task_ids}
        self.task_datasize_val = {task_id: 0 for task_id in task_ids}

        self.masked_t_loss = {task_id: 0 for task_id in task_ids}
        self.masked_v_loss = {task_id: 0 for task_id in task_ids}
        self.next_sentense_loss = {task_id: 0 for task_id in task_ids}

        self.masked_t_loss_val = {task_id: 0 for task_id in task_ids}
        self.masked_v_loss_val = {task_id: 0 for task_id in task_ids}
        self.next_sentense_loss_val = {task_id: 0 for task_id in task_ids}

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]
        del d["txt_f"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        if self.save_logger:
            self.logger = SummaryWriter(log_dir=self.log_dir)

        self.txt_f = open(self.txt_dir + "/" + "out.txt", "a")

    def txt_close(self):
        self.txt_f.close()

    def linePlot(self, step, val, split, key, xlabel="None"):
        if self.save_logger:
            self.logger.add_scalar(split + "/" + key, val, step)

    def step_train(self, epochId, stepId, loss, lr, task_id, split):
        
        self.task_loss[task_id] += loss['final_loss']
        self.task_loss_tmp[task_id] += loss['final_loss']
        self.task_lr[task_id] += lr
        self.task_lr_tmp[task_id] += lr
        self.task_step[task_id] += self.gradient_accumulation_steps
        self.task_step_tmp[task_id] += self.gradient_accumulation_steps
        self.epochId = epochId

        # plot on tensorboard.
        self.linePlot(stepId, loss['final_loss'], split, self.task_id2name[task_id] + "_final_loss")
        self.linePlot(stepId, loss['cls_loss'], split, self.task_id2name[task_id] + "_cls_loss")
        self.linePlot(stepId, loss['reg_loss'], split, self.task_id2name[task_id] + "_reg_loss")
        self.linePlot(stepId, lr, split, self.task_id2name[task_id] + "_learning_rate")

    def step_train_CC(
        self,
        epochId,
        stepId,
        masked_loss_t,
        masked_loss_v,
        next_sentence_loss,
        norm,
        task_id,
        split,
    ):
        self.masked_t_loss[task_id] += masked_loss_t
        self.masked_v_loss[task_id] += masked_loss_v
        self.next_sentense_loss[task_id] += next_sentence_loss
        self.task_norm_tmp[task_id] += norm

        self.task_step[task_id] += self.gradient_accumulation_steps
        self.task_step_tmp[task_id] += self.gradient_accumulation_steps
        self.epochId = epochId

        # plot on tensorboard.
        self.linePlot(
            stepId, masked_loss_t, split, self.task_id2name[task_id] + "_masked_loss_t"
        )
        self.linePlot(
            stepId, masked_loss_v, split, self.task_id2name[task_id] + "_masked_loss_v"
        )
        self.linePlot(
            stepId,
            next_sentence_loss,
            split,
            self.task_id2name[task_id] + "_next_sentence_loss",
        )
    def step_val(self, epochId, task_id, batch_size):
        self.task_step_val[task_id] += self.gradient_accumulation_steps
        self.task_datasize_val[task_id] += batch_size

    def step_val_CC(
        self,
        epochId,
        masked_loss_t,
        masked_loss_v,
        next_sentence_loss,
        task_id,
        batch_size,
        split,
    ):
        self.masked_t_loss_val[task_id] += masked_loss_t
        self.masked_v_loss_val[task_id] += masked_loss_v
        self.next_sentense_loss_val[task_id] += next_sentence_loss

        self.task_step_val[task_id] += self.gradient_accumulation_steps
        self.task_datasize_val[task_id] += batch_size

    def showLossValAll(self):
        progressInfo = "Eval Ep: %d " % self.epochId
        lossInfo = "Validation "
        val_scores = {}
        ave_loss = 0
        for task_id in self.task_ids:
            loss = self.task_loss_val[task_id] / float(self.task_step_val[task_id])
            score = self.task_score_val[task_id] / float(
                self.task_datasize_val[task_id]
            )
            val_scores[task_id] = score
            ave_loss += loss
            lossInfo += "[%s]: loss %.3f score %.3f " % (
                self.task_id2name[task_id],
                loss,
                score * 100.0,
            )
            self.linePlot(
                self.epochId, loss, "val", self.task_id2name[task_id] + "_loss"
            )
            self.linePlot(
                self.epochId, score, "val", self.task_id2name[task_id] + "_score"
            )
        self.task_loss_val = {task_id: 0 for task_id in self.task_loss_val}
        self.task_score_val = {task_id: 0 for task_id in self.task_score_val}
        self.task_datasize_val = {task_id: 0 for task_id in self.task_datasize_val}
        self.task_step_val = {task_id: 0 for task_id in self.task_ids}

        logger.info(progressInfo)
        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)
        return val_scores

    def getValScore(self, task_id):
        return self.task_score_val[task_id] / float(self.task_datasize_val[task_id])
    
    def getValLoss(self, task_id): 
        return self.task_cls_loss_val[task_id] / float(self.task_datasize_val[task_id])

    def showLossVal(self, task_id, curr_epoch, mAP, task_stop_controller=None):
        progressInfo = "Eval task %s on iteration %d " % (
            task_id,
            self.task_step[task_id],
        )
        lossInfo = "Validation "
        ave_loss = 0
        loss_reg = self.task_reg_loss_val[task_id] / float(self.task_datasize_val[task_id])
        loss_cls = self.task_cls_loss_val[task_id] / float(self.task_datasize_val[task_id])
        loss_final = self.task_final_loss_val[task_id] / float(self.task_datasize_val[task_id])
    
        lossInfo += "[%s]: reg_loss %.3f cls_loss %.3f final_loss %.3f " % (
            self.task_id2name[task_id],
            loss_reg, loss_cls, loss_final,
        )
        self.linePlot(
            curr_epoch, mAP, "val", self.task_id2name[task_id] + "_validation/avg_mAP"
        )
        if task_stop_controller is not None:
            self.linePlot(
                self.task_step[task_id],
                task_stop_controller[task_id].in_stop,
                "val",
                self.task_id2name[task_id] + "_early_stop",
            )
        self.task_reg_loss_val[task_id] = 0
        self.task_cls_loss_val[task_id] = 0
        self.task_final_loss_val[task_id] = 0

        self.task_score_val[task_id] = 0
        self.task_datasize_val[task_id] = 0
        self.task_step_val[task_id] = 0
        logger.info(progressInfo)
        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)
        print(lossInfo)

    def showLossTrain(self):
        # show the current loss, once showed, reset the loss.
        lossInfo = ""
        for task_id in self.task_ids:
            if self.task_num_iters[task_id] > 0:
                if self.task_step_tmp[task_id]:
                    lossInfo += (
                        "[%s]: iter %d Ep: %.2f loss %.3f lr %.6g "
                        % (
                            self.task_id2name[task_id],
                            self.task_step[task_id],
                            self.task_step[task_id]
                            / float(self.task_num_iters[task_id]),
                            self.task_loss_tmp[task_id]
                            / float(self.task_step_tmp[task_id]),
                            self.task_lr_tmp[task_id]
                            / float(self.task_step_tmp[task_id]),
                        )
                    )
        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)

        self.task_step_tmp = {task_id: 0 for task_id in self.task_ids}
        self.task_loss_tmp = {task_id: 0 for task_id in self.task_ids}
        self.task_lr_tmp = {task_id: 0 for task_id in self.task_ids}

    def showLossValCC(self):
        progressInfo = "Eval Ep: %d " % self.epochId
        lossInfo = "Validation "
        for task_id in self.task_ids:
            masked_t_loss_val = self.masked_t_loss_val[task_id] / float(
                self.task_step_val[task_id]
            )
            masked_v_loss_val = self.masked_v_loss_val[task_id] / float(
                self.task_step_val[task_id]
            )
            next_sentense_loss_val = self.next_sentense_loss_val[task_id] / float(
                self.task_step_val[task_id]
            )
            lossInfo += "[%s]: masked_t %.3f masked_v %.3f NSP %.3f" % (
                self.task_id2name[task_id],
                masked_t_loss_val,
                masked_v_loss_val,
                next_sentense_loss_val,
            )
            self.linePlot(
                self.epochId,
                masked_t_loss_val,
                "val",
                self.task_id2name[task_id] + "_mask_t",
            )
            self.linePlot(
                self.epochId,
                masked_v_loss_val,
                "val",
                self.task_id2name[task_id] + "_maks_v",
            )
            self.linePlot(
                self.epochId,
                next_sentense_loss_val,
                "val",
                self.task_id2name[task_id] + "_nsp",
            )
        self.masked_t_loss_val = {task_id: 0 for task_id in self.masked_t_loss_val}
        self.masked_v_loss_val = {task_id: 0 for task_id in self.masked_v_loss_val}
        self.next_sentense_loss_val = {
            task_id: 0 for task_id in self.next_sentense_loss_val
        }
        self.task_datasize_val = {task_id: 0 for task_id in self.task_datasize_val}
        self.task_step_val = {task_id: 0 for task_id in self.task_ids}

        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)

    def showLossTrainCC(self):
        # show the current loss, once showed, reset the loss.
        lossInfo = ""
        for task_id in self.task_ids:
            if self.task_num_iters[task_id] > 0:
                if self.task_step_tmp[task_id]:
                    lossInfo += (
                        "[%s]: iter %d Ep: %.2f masked_t %.3f masked_v %.3f NSP %.3f lr %.6g"
                        % (
                            self.task_id2name[task_id],
                            self.task_step[task_id],
                            self.task_step[task_id]
                            / float(self.task_num_iters[task_id]),
                            self.masked_t_loss[task_id]
                            / float(self.task_step_tmp[task_id]),
                            self.masked_v_loss[task_id]
                            / float(self.task_step_tmp[task_id]),
                            self.next_sentense_loss[task_id]
                            / float(self.task_step_tmp[task_id]),
                            self.task_norm_tmp[task_id]
                            / float(self.task_step_tmp[task_id]),
                        )
                    )

        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)

        self.task_step_tmp = {task_id: 0 for task_id in self.task_ids}
        self.masked_t_loss = {task_id: 0 for task_id in self.task_ids}
        self.masked_v_loss = {task_id: 0 for task_id in self.task_ids}
        self.next_sentense_loss = {task_id: 0 for task_id in self.task_ids}
        self.task_norm_tmp = {task_id: 0 for task_id in self.task_ids}

class MultiTaskStopOnPlateau(object):
    def __init__(
        self,
        mode="min",
        patience=10,
        continue_threshold=0.005,
        verbose=False,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
    ):

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.in_stop = False
        self.eps = eps
        self.last_epoch = -1
        self.continue_threshold = continue_threshold
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._init_continue_is_better(
            mode="min", threshold=continue_threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.in_stop = False

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.in_stop = True
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        # if the perforance is keep dropping, then start optimizing again.
        elif self.continue_is_better(current, self.best) and self.in_stop:
            self.in_stop = False
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        # if we lower the learning rate, then
        # call reset.

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == "min" and threshold_mode == "rel":
            rel_epsilon = 1.0 - threshold
            return a < best * rel_epsilon

        elif mode == "min" and threshold_mode == "abs":
            return a < best - threshold

        elif mode == "max" and threshold_mode == "rel":
            rel_epsilon = threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def _init_continue_is_better(self, mode, threshold, threshold_mode):

        self.continue_is_better = partial(self._cmp, mode, threshold_mode, threshold)
