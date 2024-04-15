# Unified Audio-Visual Perception for Multi-Task Video Localization
This repo is the official code of ["Unified Audio-Visual Perception for Multi-Task Video Localization"](https://arxiv.org/pdf/2404.03179.pdf).  

<!-- ## Updates
-  -->
## Introduction
This paper introduces the first unified framework to localize all three kinds of instances in untrimmed videos, including visual actions, sound events and audio-visual events. All these instances contribute equally to a comprehensive understanding of video content.
![](https://github.com/ttgeng233/UniAV/blob/main/fig1.jpg)
<!-- ![](.\overview_final_new.jpg) -->

## Requirements
The implemetation is based on PyTorch. Environment: Linux, GCC >= 4.9, CUDA >= 11.0, Python = 3.9, Pytorch = 1.11.0.  Follow [INSTALL.md](INSTALL.md) to install required dependencies.

## Data preparation
<!-- #### Download features and annotations -->
- Download ActivityNet 1.3 from [this link](https://pan.baidu.com/s/1LiRHffpyjkWYs7NR40DMjA?pwd=cjf7) (pwd: cjf7). For visual features, fps=16, sliding window size=16 and stride=8. For audio features, sample rate=16kHZ, sliding window size=1s and stride=0.5s.
- Download DESED from [this link](https://pan.baidu.com/s/1MubF5XbFMfbbumPSb-ehCA?pwd=61le) (pwd: 61le). For visual features, fps=16, sliding window size=16 and stride=4. For audio features, sample rate=16kHZ, sliding window size=1s and stride=0.25s.  
- Download UnAV-100 from [this link](https://pan.baidu.com/s/1uBRdq6mXTfnRODMbZ0-QnA?pwd=zyfm) (pwd: zyfm). For visual features, fps=16, sliding window size=16 and stride=4. For audio features, sample rate=16kHZ, sliding window size=1s and stride=0.25s.  

Details: Each link includes the files of annotations in json format and audio and visual features. The audio and visual features are extracted from the audio and visual encoder of [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE), respectively, where the visual encoder is finetuned on Kinetics-400.
<!-- #### Unpack features and annotations -->
After downloading, unpack files under `./data`. The folder structure should look like:
```
This folder
│   README.md
│   ...  
└───data/
│    └───activitynet13/
│    |	 └───annotations
│    |	 └───av_features  
│    └───desed/
│    |	 └───annotations
│    |	 └───av_features 
│    └───unav100/
│    	 └───annotations
│    	 └───av_features  
└───libs
│   ...
```
## Training and evaluation
Coming soon.

## Running on your own videos
Coming soon.

<!-- ## Evaluation
Coming soon. -->


## Citation
If you find our data and code are useful for your research, please consider citing our paper
```
@article{geng2024uniav,
  title={UniAV: Unified Audio-Visual Perception for Multi-Task Video Localization},
  author={Geng, Tiantian and Wang, Teng and Zhang, Yanfu and Duan, Jinming and Guan, Weili and Zheng, Feng},
  journal={arXiv preprint arXiv:2404.03179},
  year={2024}
}
```

## Acknowledgement
The video and audio features were extracted using [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE). Our baseline model was implemented based on [ActionFormer](https://github.com/happyharrycn/actionformer_release) and [UnAV](https://github.com/ttgeng233/UnAV). We thank the authors for their efforts. If you use our code, please also consider to cite their works.