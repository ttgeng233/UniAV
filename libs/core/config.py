import yaml


DEFAULTS = {
    # random seed for reproducibility, a large number is preferred
    "init_rand_seed": 1234567891, 
    "devices": ['cuda:0'], # default: single gpu
    "model_name": "LocPointTransformer",
    "multi_modal": True,  
    "output_folder": "./ckpt", ##file path to save checkpoint
    "train_iter_gap": 4, 
    "num_workers": 2,
    "dataset": {
        # downsampling rate of features, 1 to use original resolution
        "downsample_rate": 1,
        # set to a tuple (e.g., (0.9, 1.0)) to enable random feature cropping
        "crop_ratio": [0.9, 1.0],
        "file_prefix": "",
        "file_ext": ".npy",
        "force_upsampling": True,
    },
    # network architecture
    "model": {
        "backbone_type": 'convTransformer',
        "input_dim_V": 1536, 
        "input_dim_A": 1536,  
        "backbone_arch": (2, 3, 5),
        # scale factor between pyramid levels
        "scale_factor": 2,
        # regression range for pyramid levels
        "regression_range": [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        # number of heads in self-attention
        "n_head": 4,
        # kernel size for embedding network
        "embd_kernel_size": 3,
        # (output) feature dim for embedding network
        "embd_dim": 512,  
        # if attach group norm to embedding network
        "embd_with_ln": True,
        # feat dim for head
        "head_dim": 512,  
        # kernel size for reg/cls/center heads
        "head_kernel_size": 3,
        # number of layers in the head (including the final one)
        "head_num_layers": 3,
        # if attach group norm to heads
        "head_with_ln": True,
        # disable abs position encoding (added to input embedding)
        "use_abs_pe": True, 
    },
    "train_cfg": {
        # on reg_loss, use -1 to enable auto balancing
        "loss_weight": 2, 
        # use prior in model initialization to improve stability
        "cls_prior_prob": 0.01,
        "init_loss_norm": 250,  
        # gradient cliping, not needed for pre-LN transformer
        "clip_grad_l2norm": 1.0,
        # cls head without data 
        "head_empty_cls": [],  
        # dropout ratios for tranformers
        "dropout": 0.0,
        # ratio for drop path
        "droppath": 0.1,
        # if to use label smoothing (>0.0)
        "label_smoothing": 0.0,
        "evaluate": True, 
        "eval_freq": 1,
    },
    "test_cfg": {
        "pre_nms_thresh": 0.001,
        "pre_nms_topk": 2000,
        "iou_threshold": 0.7,
        "min_score": 0.001,
        "max_seg_num": 100,
        "nms_method": 'soft', # soft | hard | none
        "nms_sigma" : 0.4, 
        "duration_thresh": 0.05,
        "multiclass_nms": True,
        "ext_score_file": None,
        "voting_thresh" : 0.75,
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "AdamW", # SGD or AdamW
        # solver params
        "momentum": 0.9,
        "weight_decay": 0.0001, 
        "learning_rate": 1e-4, 
        # excluding the warmup epochs
        "epochs": 40,  
        # lr scheduler: cosine / multistep
        "warmup": True,
        "warmup_epochs": 2, 
        "schedule_type": "cosine",
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
        "clip_proj_lr": 1e-3,  
    }
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def load_default_config():
    config = DEFAULTS
    return config

def _update_config(config, task_config):
    # fill in derived fields
    config["dataset"]["multi_modal"] = config["multi_modal"]  
    config["dataset"]["backbone_arch"] = config["model"]["backbone_arch"]
    config["dataset"]["regression_range"] = config["model"]["regression_range"]
    config["dataset"]["scale_factor"] = config["model"]["scale_factor"]
    
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]

    _merge(config["dataset"], task_config["TASK1"]['dataset'])
    _merge(config["dataset"], task_config["TASK2"]['dataset'])
    _merge(config["dataset"], task_config["TASK3"]['dataset']) 

    config["model"]["task_cfg"] = task_config

    return task_config

def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        task_config = yaml.load(fd, Loader=yaml.FullLoader)
    config = load_default_config()
    task_config = _update_config(config, task_config)
    return config, task_config
