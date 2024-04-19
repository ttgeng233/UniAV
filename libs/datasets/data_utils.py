import os
import copy
import random
import numpy as np
import random
import torch
from torch.nn import functional as F


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def truncate_feats(
    data_dict,
    max_seq_len,
    trunc_thresh,
    offset, 
    crop_ratio=None,
    multi_modal = True,
    max_num_trials=200,
    has_action=True,
    no_trunc=False

):
    """
    Truncate feats and time stamps in a dict item

    data_dict = {'video_id'        : str
                 'feats'           : Tensor C x T
                 'segments'        : Tensor N x 2 (in feature grid)
                 'labels'          : Tensor N
                 'fps'             : float
                 'feat_stride'     : int
                 'feat_num_frames' : in

    """
    # get the meta info
    if multi_modal:
        feat_len = data_dict['feats']['visual'].shape[1]
    else:
        feat_len = data_dict['feats'].shape[1]

    num_segs = data_dict['segments'].shape[0]

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        # do nothing
        if crop_ratio == None:
            return data_dict
        # randomly crop the seq by setting max_seq_len to a value in [l, r]
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # # corner case
            if feat_len == max_seq_len:
                return data_dict

    # otherwise, deep copy the dict
    data_dict = copy.deepcopy(data_dict)

    # try a few times till a valid truncation with at least one action
    for _ in range(max_num_trials):

        # sample a random truncation of the video feats
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        window = torch.as_tensor([st, ed], dtype=torch.float32)

        # compute the intersection between the sampled window and all segments
        window = window[None].repeat(num_segs, 1)
        left = torch.maximum(window[:, 0] - offset, data_dict['segments'][:, 0]) 
        right = torch.minimum(window[:, 1] + offset, data_dict['segments'][:, 1])  
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(
            data_dict['segments'][:, 1] - data_dict['segments'][:, 0])
        inter_ratio = inter / area_segs

        # only select those segments over the thresh
        seg_idx = (inter_ratio >= trunc_thresh)

        if no_trunc:
            # with at least one action and not truncating any actions
            seg_trunc_idx = torch.logical_and(
                (inter_ratio > 0.0), (inter_ratio < 1.0)
            )
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if seg_idx.sum().item() > 0:
                break
        else:
            # without any constraints
            break
    
    if multi_modal:
        data_dict['feats']['visual'] = data_dict['feats']['visual'][:, st:ed].clone()
        data_dict['feats']['audio'] = data_dict['feats']['audio'][:, st:ed].clone()
    else:
        # feats: C x T
        data_dict['feats'] = data_dict['feats'][:, st:ed].clone()
    # segments: N x 2 in feature grids
    data_dict['segments'] = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    # shift the time stamps due to truncation
    data_dict['segments'] = data_dict['segments'] - st
    # labels: N
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()

    return data_dict


def label_points(points, 
                 gt_segments, 
                 gt_labels, 
                 num_classes, 
                 class_aware
                 ):
        # concat points on all pyramid levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        concat_points = torch.cat(points, dim=0)
        cls_targets, reg_targets = label_points_single_video(
            concat_points, gt_segments, gt_labels, num_classes, class_aware)
        return cls_targets, reg_targets

def label_points_single_video(concat_points, 
                              gt_segment, 
                              gt_label, 
                              num_classes, 
                              class_aware
                              ):
        # concat_points : F T x 4 (t, regressoin range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0] 
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        # inside an gt action
        inside_gt_seg_mask = reg_targets.min(-1)[0] > 0 

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        lens.masked_fill_(inside_gt_seg_mask==0, float('inf')) 
        lens.masked_fill_(inside_regress_range==0, float('inf'))

        if class_aware:
            min_len  = lens
            min_len_mask = (min_len < float('inf')).to(reg_targets.dtype) 
        else:
            # if there are still more than one actions for one moment
            # pick the one with the shortest duration (easiest to regress)
            # F T x N -> F T
            min_len, min_len_inds = lens.min(dim=1)
            # corner case: multiple actions with very similar durations 
            min_len_mask = torch.logical_and(
                (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
            ).to(reg_targets.dtype) 

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        
        if class_aware:
            new_reg_targets = torch.zeros(num_pts, num_classes, 2)
            for i in range(num_pts):
                inds = min_len_mask[i].nonzero() 
                new_reg_targets[i, gt_label[inds]] = reg_targets[i, inds] 
            new_reg_targets /= concat_points[:, 3, None, None]
        else:        
            # OK to use min_len_inds
            new_reg_targets = reg_targets[range(num_pts), min_len_inds] 
            # normalization based on stride
            new_reg_targets /= concat_points[:, 3, None]

        return cls_targets, new_reg_targets