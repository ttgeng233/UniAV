import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch import nn

from .datasets import register_dataset, make_generator
from .data_utils import truncate_feats, label_points

@register_dataset("unav100")
class UnAV100Dataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        scale_factor,          # scale factor between branch layers
        regression_range,      # regression range on each level of FPN
        backbone_arch,         # a tuple defines # layers in embed / stem / branch
        class_aware,            # if use class-aware regression
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling, 
        multi_modal
    ):
        self.multi_modal = multi_modal
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

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
        # "empty" noun categories on epic-kitchens
        assert len(label_dict) <= num_classes
        self.data_list = dict_db
        self.label_dict = label_dict
        self.class_list = class_list 

        # dataset specific attributes
        empty_label_ids = self.find_empty_cls(label_dict, num_classes)
        self.db_attributes = {
            'dataset_name': 'unav-100',
            'tiou_thresholds': np.linspace(0.1, 0.9, 9), 
            'empty_label_ids': empty_label_ids
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

    def find_empty_cls(self, label_dict, num_classes):
        # find categories with out a data sample
        if len(label_dict) == num_classes:
            return []
        empty_label_ids = []
        label_ids = [v for _, v in label_dict.items()]
        for id in range(num_classes):
            if id not in label_ids:
                empty_label_ids.append(id)
        return empty_label_ids

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
            class_prompt = 'An audio visual event of ' + clss + '.'
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

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                num_acts = len(value['annotations'])
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(value['annotations']):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
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
        # directly return a (truncated) data point 
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]
        filename_rgb = os.path.join(self.feat_folder,
                        self.file_prefix + video_item['id'] 
                        + '_one_peace_video_finetune' + self.file_ext)
        feats_rgb = np.load(filename_rgb).astype(np.float32)
        feats_visual = feats_rgb
        # deal with downsampling (= increased feat stride)
        feats_visual = feats_visual[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate

        # T x C -> C x T
        feats_visual = torch.from_numpy(np.ascontiguousarray(feats_visual.transpose()))

        if self.multi_modal:
            #load audio features
            filename_audio = os.path.join(self.feat_folder,
                            self.file_prefix + video_item['id'] 
                            + '_one_peace_audio' + self.file_ext)
            feats_audio = np.load(filename_audio).astype(np.float32)
            feats_audio = feats_audio[::self.downsample_rate, :]
            # T x C -> C x T
            feats_audio = torch.from_numpy(np.ascontiguousarray(feats_audio.transpose()))

            #avoid audio and visual features have different lengths
            min_len = min(feats_visual.shape[1], feats_audio.shape[1])
            feats = {'visual': feats_visual[:, :min_len], 'audio': feats_audio[:, :min_len] }
        else: 
            feats = feats_visual

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                (video_item['segments'] * video_item['fps']- 0.5 * self.num_frames) / feat_stride
            )
            labels = torch.from_numpy(video_item['labels'])
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
                     'feat_num_frames' : self.num_frames}

        # truncate the features during training
        feat_offset = 0.5 * self.num_frames / feat_stride 
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, 
                self.crop_ratio, self.multi_modal
            )

        # compute the gt labels for cls & reg
        gt_segments = data_dict['segments']
        gt_labels = data_dict['labels']
        if self.multi_modal:
            points = self.point_generator(self.fpn_strides, 
                                      data_dict['feats']['visual'], 
                                      self.is_training) 
        else:
            points = self.point_generator(self.fpn_strides, 
                                        data_dict['feats'], 
                                        self.is_training)  
        data_dict['gt_cls_labels'], data_dict['gt_offsets'] = label_points(
                points, gt_segments, gt_labels, self.num_classes, self.class_aware)
        data_dict['points'] = points

        return data_dict
