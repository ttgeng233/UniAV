import os
import json
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset, make_generator
from .data_utils import truncate_feats, label_points
from ..utils import remove_duplicate_annotations

@register_dataset("anet")
class ActivityNetDataset(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        feat_folder,      # folder for features
        json_file,        # json file for annotations
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats
        max_seq_len,      # maximum sequence length during training
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        scale_factor,          # scale factor between branch layers
        regression_range,      # regression range on each level of FPN
        backbone_arch,         # a tuple defines # layers in embed / stem / branch
        class_aware,            # if use class-aware regression
        trunc_thresh,     # threshold for truncate an action segment
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        num_classes,      # number of action categories
        file_prefix,      # feature file prefix if any
        file_ext,         # feature file extension if any
        force_upsampling,  # force to upsample to max_seq_len
        multi_modal,
    ):
        self.multi_modal = multi_modal
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        self.use_hdf5 = '.hdf5' in feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict, class_list = self._load_json_db(self.json_file)
        # proposal vs action categories
        assert (num_classes == 1) or (len(label_dict) == num_classes)
        self.data_list = dict_db
        self.label_dict = label_dict
        self.class_list = class_list

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'ActivityNet 1.3',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10),
            'empty_label_ids': []
        }

        # location generator: points
        self.fpn_strides = [scale_factor**i for i in range(backbone_arch[-1]+1)]
        self.reg_range = regression_range
        self.class_aware = class_aware
        max_div_factor = 1
        for l, stride in enumerate(self.fpn_strides):
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len_ori' : self.max_seq_len,
                'max_buffer_len_factor': max_buffer_len_factor, 
                'fpn_levels' : len(self.fpn_strides),
                'scale_factor' : scale_factor,
                'regression_range' : self.reg_range,
                'max_div_factor' : self.max_div_factor  
            }
        )

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        class_list = []
        label_dict_sorted = sorted(label_dict.items(), key=lambda item : item[1])
        for clss, id in label_dict_sorted:
            class_prompt = 'A visual event of ' + clss + '.' 
            class_list.append(class_prompt)

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."
            duration = value['duration']

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                valid_acts = remove_duplicate_annotations(value['annotations'])
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(valid_acts):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    if self.num_classes == 1:
                        labels[idx] = 0
                    else:
                        labels[idx] = label_dict[act['label']]
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict, class_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        if self.use_hdf5:
            with h5py.File(self.feat_folder, 'r') as h5_fid:
                feats = np.asarray(
                    h5_fid[self.file_prefix + video_item['id']][()],
                    dtype=np.float32
                )
        else:
            filename_rgb = os.path.join(self.feat_folder,
                        self.file_prefix + video_item['id'] 
                        + '_one_peace_video_finetune' + self.file_ext)
            feats_rgb = np.load(filename_rgb).astype(np.float32)
            feats_visual = feats_rgb
        if self.multi_modal:
            filename_audio = os.path.join(self.feat_folder,
                            self.file_prefix + video_item['id'] 
                            + '_one_peace_audio' + self.file_ext)
            feats_audio = np.load(filename_audio).astype(np.float32)
            #avoid audio and visual features have different lengths
            min_len = min(feats_visual.shape[0], feats_audio.shape[0])
            feats_visual = feats_visual[:min_len, :]
            feats_audio = feats_audio[:min_len, :]
        else:
            feats = feats_visual

        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling):
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                feats = feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate
        # case 2: variable length features for input, yet resized for training
        elif self.feat_stride > 0 and self.force_upsampling:
            feat_stride = float(
                (feats_visual.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = feats.shape[0]
            assert seq_len <= self.max_seq_len
            if self.force_upsampling:
                # reset to max_seq_len
                seq_len = self.max_seq_len
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len
            # center the features
            num_frames = feat_stride
        feat_offset = 0.5 * num_frames / feat_stride

        # resize the features if needed
        if self.multi_modal:
            feats_visual = torch.from_numpy(np.ascontiguousarray(feats_visual.transpose()))
            feats_audio = torch.from_numpy(np.ascontiguousarray(feats_audio.transpose()))
            feats = {'visual': feats_visual, 'audio': feats_audio}
            if (feats_visual.shape[-1] != self.max_seq_len) and self.force_upsampling:
                resize_feats_visual = F.interpolate(
                    feats_visual.unsqueeze(0),
                    size=self.max_seq_len,
                    mode='linear',
                    align_corners=False
                )
                resize_feats_audio = F.interpolate(
                    feats_audio.unsqueeze(0),
                    size=self.max_seq_len,
                    mode='linear',
                    align_corners=False
                )
                feats_visual = resize_feats_visual.squeeze(0)
                feats_audio = resize_feats_audio.squeeze(0)
                feats = {'visual': feats_visual, 'audio': feats_audio}
        else:
            # T x C -> C x T
            feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
            if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
                resize_feats = F.interpolate(
                    feats.unsqueeze(0),
                    size=self.max_seq_len,
                    mode='linear',
                    align_corners=False
                )
                feats = resize_feats.squeeze(0)

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
            # for activity net, we have a few videos with a bunch of missing frames
            # here is a quick fix for training
            if self.is_training:
                if self.multi_modal:
                    vid_len = feats['visual'].shape[1] + feat_offset  
                else:
                    vid_len = feats.shape[1] + feat_offset

                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= vid_len:
                        # skip an action outside of the feature map
                        continue
                    # skip an action that is mostly outside of the feature map
                    if seg[1].item() - seg[0].item() == 0.0:
                        print(0)
                    ratio = (
                        (min(seg[1].item(), vid_len) - seg[0].item())
                        / (seg[1].item() - seg[0].item())
                    )
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len))
                        # some weird bug here if not converting to size 1 tensor
                        valid_label_list.append(label.view(1))
                if len(valid_seg_list)==0:
                    print(0)
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames}

        # no truncation is needed
        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, 
                self.crop_ratio, self.multi_modal
            )
        
        # compute the gt labels for cls & reg
        gt_segments = data_dict['segments']
        gt_labels = data_dict['labels']
        if self.multi_modal:
            points = self.point_generator(self.fpn_strides, data_dict['feats']['visual'], self.is_training)
        else:
            points = self.point_generator(self.fpn_strides, data_dict['feats'], self.is_training)  

        data_dict['gt_cls_labels'], data_dict['gt_offsets'] = label_points(
                points, gt_segments, gt_labels, self.num_classes, self.class_aware)
        data_dict['points'] = points

        return data_dict