import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import (register_multimodal_meta_arch, make_multimodal_backbone)
from .blocks import MaskedConv1D, Scale, LayerNorm, Linear
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss

from ..utils import batched_nms

import numpy as np


class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = [],
        clip_dim = 1536,
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()

        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.clip_proj = Linear(clip_dim, feat_dim) 
        self.vis_proj = MaskedConv1D(
                            feat_dim, feat_dim, 
                            kernel_size, stride=1,
                            padding=kernel_size//2)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, fpn_feats, fpn_masks, clip_class_feature, task_id):
        assert len(fpn_feats) == len(fpn_masks)
        # apply the classifier for each pyramid level
        out_logits = tuple()
        clip_class_feature = clip_class_feature[task_id].type(fpn_feats[0].dtype)
        clip_class_feature = self.clip_proj(clip_class_feature)
        clip_class_feature = F.normalize(clip_class_feature, dim=1) 
        logit_scale = self.logit_scale.exp()

        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            #clip  
            cur_out, _ = self.vis_proj(cur_out, cur_mask)
            cur_out = F.normalize(cur_out, dim=1)   
            cur_logits = logit_scale * cur_out.permute(0, 2, 1) @ clip_class_feature.t()
            cur_logits = cur_logits * cur_mask.permute(0, 2, 1).type(cur_out.dtype)
            out_logits += (cur_logits, )

        return out_logits

class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        class_aware=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()
        self.num_classes = num_classes
        self.class_aware = class_aware

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim

            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        self.offset_head = nn.ModuleDict()
        for task in self.class_aware.keys():
            if self.class_aware[task]:
                self.offset_head[task] = MaskedConv1D(
                    feat_dim, 2*num_classes[task], kernel_size,
                    stride=1, padding=kernel_size//2
                )
            else:
                self.offset_head[task] = MaskedConv1D( 
                    feat_dim, 2, kernel_size,
                    stride=1, padding=kernel_size//2
                )

    def forward(self, fpn_feats, fpn_masks, task_id):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))

            cur_offsets, _ = self.offset_head[task_id](cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), ) 

        return out_offsets


@register_multimodal_meta_arch("LocPointTransformer")
class PtTransformer(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        backbone_type,         # a string defines backbone type
        backbone_arch,         # a tuple defines # layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim_V,           # input visual feat dim
        input_dim_A,           # input audio feat dim
        n_head,                # number of heads for self-attention in transformer
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        train_cfg,             # other cfg for training
        test_cfg,              # other cfg for testing
        task_cfg,              # task-specific cfg
    ):
        super().__init__()
        self.fpn_strides = [scale_factor**i for i in range(backbone_arch[-1]+1)]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor

        class_aware = {} 
        max_seq_len = {}
        num_classes = {}
        train_loss_weight = {} 
        train_label_smoothing = {}
        loss_normalizer = {}

        class_list = {}
        clip_class_feature = {}
        max_div_factor = 1
        for task in task_cfg.keys():  
            max_seq_len[task] = task_cfg[task]['dataset']['max_seq_len']
            for l, stride in enumerate(self.fpn_strides):
                assert max_seq_len[task] % stride == 0, "max_seq_len must be divisible by fpn stride"
                if max_div_factor < stride: 
                    max_div_factor = stride   
        self.max_div_factor = max_div_factor
        self.max_seq_len = max_seq_len 
        for task in task_cfg.keys(): 
            class_aware[task] = task_cfg[task]['dataset']['class_aware']
            num_classes[task] = task_cfg[task]['dataset']['num_classes']
            train_loss_weight[task] = task_cfg[task]['train_cfg']['loss_weight']
            train_label_smoothing[task] = task_cfg[task]['train_cfg']['label_smoothing']
            loss_normalizer[task] = task_cfg[task]['train_cfg']['init_loss_norm']
            # class_list[task] = task_cfg[task]['class_list'] 
            clip_class_feature[task] = task_cfg[task]['clip_class_feature']

        self.class_aware = class_aware 
        self.num_classes = num_classes
        self.train_loss_weight = train_loss_weight
        self.train_label_smoothing = train_label_smoothing
        self.loss_normalizer = loss_normalizer
        # self.class_list = class_list
        self.clip_class_feature = clip_class_feature

        # training time config
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']

        test_multiclass_nms = {}
        test_nms_sigma = {}
        for task in task_cfg.keys():
            test_multiclass_nms[task] = task_cfg[task]['test_cfg']['multiclass_nms']
            test_nms_sigma[task] = task_cfg[task]['test_cfg']['nms_sigma']
        self.test_multiclass_nms = test_multiclass_nms 
        self.test_nms_sigma = test_nms_sigma
        self.test_voting_thresh = test_cfg['voting_thresh']

        # backbone network: conv + transformer
        assert backbone_type in ['convTransformer']
        self.backbone = make_multimodal_backbone(
            'convTransformer',
            **{
                'n_in_V' : input_dim_V,
                'n_in_A' : input_dim_A,
                'n_embd' : embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': self.max_seq_len,  
                'arch' : backbone_arch,
                'scale_factor' : scale_factor,
                'with_ln' : embd_with_ln,
                'attn_pdrop' : 0.0,
                'proj_pdrop' : self.train_dropout,
                'path_pdrop' : self.train_droppath,
                'use_abs_pe' : use_abs_pe,
            }
        )
        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            embd_dim*2,
            head_dim, self.num_classes,  
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            embd_dim*2,
            head_dim, self.num_classes, 
            len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln,
            class_aware=self.class_aware
        )
        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, video_list, task_id, task_type):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs_V, batched_inputs_A, batched_masks = self.preprocessing(video_list, task_id)

        # forward the backbone
        feats_V, feats_A, masks = self.backbone(batched_inputs_V, batched_inputs_A,
                                                 batched_masks, task_id, task_type)

        #concat audio and visual output features (B, C, T)->(B, 2C, T)
        feats_AV = [torch.cat((V, A), 1) for _, (V, A) in enumerate(zip(feats_V, feats_A))]

        out_cls_logits = self.cls_head(feats_AV, masks, self.clip_class_feature, task_id)
        out_offsets = self.reg_head(feats_AV, masks, task_id)

        if self.class_aware[task_id]:
            out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
            out_offsets = [x.view(x.shape[0], x.shape[1], self.num_classes[task_id], -1).contiguous() for x in out_offsets]
        else:
            # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
            out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in masks]

        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_offsets = [x['gt_offsets'] for x in video_list]
            gt_cls_labels = [x['gt_cls_labels'] for x in video_list]

            # compute the loss and return
            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets, task_id
            )
            return losses

        else:
            results = self.inference(
                video_list, fpn_masks,
                out_cls_logits, out_offsets, task_id
            )
            return results

    @torch.no_grad()
    def preprocessing(self, video_list, task_id, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats_visual = [x['feats']['visual'] for x in video_list]
        feats_audio = [x['feats']['audio'] for x in video_list]
        feats_lens = torch.as_tensor([feat_visual.shape[-1] for feat_visual in feats_visual])
        max_len = feats_lens.max(0).values.item() 

        if self.training:
            assert max_len <= self.max_seq_len[task_id], "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len[task_id]
        else:
            if max_len <= self.max_seq_len[task_id]:
                max_len = self.max_seq_len[task_id]
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride

        # batch input shape B, C, T->visual
        batch_shape_visual = [len(feats_visual), feats_visual[0].shape[0], max_len]
        batched_inputs_visual = feats_visual[0].new_full(batch_shape_visual, padding_val)
        for feat_visual, pad_feat_visual in zip(feats_visual, batched_inputs_visual):
            pad_feat_visual[..., :feat_visual.shape[-1]].copy_(feat_visual)

        # audio 
        batch_shape_audio = [len(feats_audio), feats_audio[0].shape[0], max_len]
        batched_inputs_audio = feats_audio[0].new_full(batch_shape_audio, padding_val)
        for feat_audio, pad_feat_audio in zip(feats_audio, batched_inputs_audio):
            pad_feat_audio[..., :feat_audio.shape[-1]].copy_(feat_audio) 
   
        # generate the mask 
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]
        # push to device
        batched_inputs_visual = batched_inputs_visual.to(self.device)
        batched_inputs_audio = batched_inputs_audio.to(self.device)
        
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs_visual, batched_inputs_audio, batched_masks

    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets, task_id
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        
        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask] 
        gt_offsets = torch.stack(gt_offsets)[pos_mask] 

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer[task_id] = self.loss_normalizer_momentum * self.loss_normalizer[task_id] + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing[task_id]
        gt_target += self.train_label_smoothing[task_id] / (self.num_classes[task_id] + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer[task_id]

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum',
                class_aware=self.class_aware[task_id]
            )
            reg_loss /= self.loss_normalizer[task_id]

        if self.train_loss_weight[task_id] > 0:
            loss_weight = self.train_loss_weight[task_id]
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                'final_loss' : final_loss}

    @torch.no_grad()
    def inference(
        self,
        video_list,
        fpn_masks,
        out_cls_logits, out_offsets,
        task_id
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]
        vid_points = [x['points'] for x in video_list]  

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes, points) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes, vid_points)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid, task_id
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # step 3: postprocssing
        # results = self.postprocessing(results)
        results = self.postprocessing(results, task_id)


        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
        task_id
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks
            ):
            # sigmoid normalization for output logits
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1] 
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0] 

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs =  torch.div(
                topk_idxs, self.num_classes[task_id], rounding_mode='floor'
            ) 
            cls_idxs = torch.fmod(topk_idxs, self.num_classes[task_id]) 
            
            if self.class_aware[task_id]:
                # 3. gather predicted offsets
                offsets_i = offsets_i.view(-1, offsets_i.shape[-1]).contiguous() 
                offsets = offsets_i[topk_idxs] 

            else:
                # 3. gather predicted offsets
                offsets = offsets_i[pt_idxs] 

            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1) 

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all}

        return results

    @torch.no_grad()
    # def postprocessing(self, results):
    def postprocessing(self, results, task_id):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration'] 
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms = (self.test_nms_method == 'soft'),
                    multiclass = self.test_multiclass_nms[task_id],
                    sigma = self.test_nms_sigma[task_id],
                    voting_thresh = self.test_voting_thresh
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
            # 4: repack the results
            processed_results.append(
                {'video_id' : vidx,
                 'segments' : segs,
                 'scores'   : scores,
                 'labels'   : labels}
            )

        return processed_results
