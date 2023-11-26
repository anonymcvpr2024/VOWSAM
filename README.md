# VOWSAM: Rethinking Feature Aggregation in Video Object Detection with SAM


<p align="center">
  <span style="display: inline-block; text-align: center; width: 48%;">
    <img src="assets/intro_figure_direction.jpg" width="100%" style="vertical-align: top;" />
    <br><br>
    <Strong>Introducing Novel Instance Mask-based Feature Aggregation Paradigm in Video Object Detection</Strong>
  </span>
  <img src="assets/sota_comparison.png" width="48%" style="display: inline-block; vertical-align: top;" />
</p>

<p align="center">
  <img src="assets/VOD_MOT.gif" alt="VOD MOT Demo" width="100%" />
    <br>
    <Strong>Our demo results on Video Object Detection with VOWSAM (YOLOX) and Multi-Object Tracking with ByteTrack</Strong>
</p>

## Main Results of VOWSAM 
YOLOV checkpoint can be accessed from [Here](https://github.dev/YuHengsss/YOLOV)

| Model                                                                                                                | mAP@50<sup>val<br> | Speed 2080Ti(batch size=1)<br>(ms) | Config                                                                                       |                                                   Checkpoint                                                   |
|----------------------------------------------------------------------------------------------------------------------|:------------------:|:----------------------------------:|-----------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------:|
| YOLOV-S                                                                                                              |        77.3        |                11.3                | [Config](exps/yolov/yolov_s.py)                                                              |                                                                                                                |
| VOWSAM-S                                                                                                             |        78.2        |                11.6                | [Config](exps/vowsam/vowsam_s.py)                                                            | [Anonymous drive link ](https://drive.google.com/file/d/1OU2NaNOyiQLc8KS1Qp-JN8FPrmwRmo8X/view?usp=drive_link) |
| YOLOV-L                                                                                                              |        83.6        |                16.4                | [Config](exps/yolov/yolov_l.py)                                                              |                                                                                                                |
| VOWSAM-L                                                                                                             |        84.3        |                16.5                | [Config](exps/vowsam/vowsam_l.py)                                                            |     [Anonymous drive link ](https://drive.google.com/file/d/1GWr8ghwZeDnrz5V3vqYVxF-XQ8fnzYZH/view?usp=drive_link)                                                                                                           |
| YOLOV-X                                                                                                              |        85.5        |                22.7                | [Config](exps/yolov/yolov_x.py)                                                              |                                                                                                                |
| VOWSAM-X                                                                                                             |        85.6        |                22.7                | [Config](exps/vowsam/vowsam_x.py)                                                            |           [Anonymous drive link ](https://drive.google.com/file/d/1UjCyhWDuLIz6wmu0TApBMD5NVN51N0jx/view?usp=drive_link)                                                                                                     |
| YOLOV-X + [post](https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection)  |        87.5        |                 -                  | [Config](exps/yolov/yolov_x.py)                                                         |                                                       -                                                        |
| VOWSAM-X + [post](https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection) |        87.9        |                 -                  | [Config](exps/vowsam/vowsam_x.py)                                                       |                                                       -                                                        |



## Results on Multi-Object Tracking
### Results with Tracktor As Baseline
|      Method       | MOTA | IDF1 | FP | FN | IDSw. | Config |                                                                    Checkpoint                                                                    |
|:-----------------:| :--: | :--: |:--:|:--:| :---: | :----: |:------------------------------------------------------------------------------------------------------------------------------------------------:|
|     Tracktor      | 70.5 | 65.3 | 3659 | 176118 | 1442 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot15-public-half.py) | [Link from MMtracking](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-half_20210804_001040-ae733d0c.pth) |
| **Tracktor+Ours** | 71.4 | 66.7 | 3419 | 171174 | 1344 | [config](tracktor_faster-rcnn_r50_fpn_4e_mot15-private-half.py) |                  [Anonymous drive link](https://drive.google.com/file/d/1X-6L0KcWUe0smq6cML8_m8XbrZe-LhXY/view?usp=drive_link)                   |

### Results with ByteTrack As Baseline
|       Method       | HOTA | MOTA | IDF1 | FP | FN | IDSw. | Config | Checkpoint |
|:------------------:| :--: | :--: | :--: |:--:|:--:| :---: | :----: | :--------: |
|     ByteTrack      | 65.5 | 86.4 | 82.7 | 19176 | 63370 | 995 | [config](bytetrack_yolox_x_crowdhuman_mot17-private-half.py) | [Anonymous drive link](https://drive.google.com/file/d/12X_gqf7LcXUE8hw22k01xWb8LRjyEqP7/view?usp=drive_link) |
| **ByteTrack+Ours** | 68.9 | 88.1 | 83.7 | 18647 | 53825 | 911 | [config](bytetrack_yolox_x_crowdhuman_mot17-private.py) | [Anonymous drive link](https://drive.google.com/file/d/1y7ZhGEHxCrhtF12yZDY-M2KXOQ2eOaIe/view?usp=drive_link) |




## Quick Start on VOWSAM

<details>
<summary>Installation</summary>

Install VOWSAM from source.
```shell
git clone git@github.com:anonymcvpr2024/VOWSAM.git
cd VOWSAM
```

Create conda env.
```shell
conda create -n vowsam python=3.7

conda activate vowsam

pip install -r requirements.txt

pip3 install -v -e .
```
</details>


<details>
<summary>Reproduce our results on VID</summary>

Step1. Download datasets and weights:

Download ILSVRC2015 DET and ILSVRC2015 VID dataset from [IMAGENET](https://image-net.org/challenges/LSVRC/2015/2015-downloads) and organise them as follows:

```shell
path to your datasets/ILSVRC2015/
path to your datasets/ILSVRC/
```

Download the COCO-style annotations from YOLOV for [training](https://drive.google.com/file/d/1HhE4OAcc--CpjUj69JCRXzMvIRsR4ymM/view?usp=sharing) and [video sequences](https://drive.google.com/file/d/1vJs8rLl_2oZOWCMJtk3a9ZJmdNn8cu-G/view?usp=sharing). Then, put them in these two directories:
```shell
annotations/vid_train_coco.json
yolox/data/dataset/train_seq.npy
```

Change the data_dir in exp files to [path to your datasets] and Download our checkpoints.

Step2. Generate predictions and convert them to IMDB style for evaluation.

```shell
python tools/val_to_imdb.py -f exps/vowsam/vowsam_s.py -c path to your weights/vowsam_s.pth --fp16 --output_dir ./vowsam_s.pkl
```
Evaluation process:
```shell
python tools/REPPM.py --repp_cfg ./tools/yolo_repp_cfg.json --predictions_file ./vowsam_s.pkl --evaluate --annotations_filename ./annotations/annotations_val_ILSVRC.txt --path_dataset [path to your dataset] --store_imdb --store_coco  (--post)
```
(--post) indicates involving post-processing method.

  
**Training example**
```shell
python tools/vid_train.py -f exps/vowsam/vowsam_s.py -c weights/yoloxs_vid.pth --fp16
```
**Roughly testing**
```shell
python tools/vid_eval.py -f exps/yolov/vowsam_s.py -c weights/vowsam_s.pth --tnum 500 --fp16
```
tnum indicates testing sequence number.
</details>


<details>
<summary>Generating GT Instance Masks from SAM</summary>


**For ImageNet VID dataset**

Experiment file is needed to set other parameters such as data set path.
```shell
python tools/generate_masks_from_SAM.py -f exps/vowsam/vowsam_s.py -c [path to SAM checkpoint]
```

**For ImageNet DET dataset**

```shell
python tools/generate_masks_from_SAM_DET.py -f exps/vowsam/vowsam_s.py -c [path to SAM checkpoint]
```
</details>

<details>
<summary>Visualization</summary>

* **Visualizing Generated SAM Mask and corresponding GT on the training set**

    ```shell
    python tools/visualize_vid_train.py -f [experiment_name_file]
    ```
  

* **Visualizing Predictions (bbox, class label) on any video**

  The script takes experiment  file, checkpoint,  and path to video as mp4 and writes a Video (.mp4) or images in the same directory where model checkpoint is located.
    ```shell
    python tools/vid_demo.py -f [experiment_name_file] -c ..[checkpoint_path] --path [video_file]
    ```
</details>


## For Experiments on Multi-Object Tracking go to [MMtracking Folder](MMtracking)


## Acknowledgements

<details><summary> <b>Expand</b> </summary>
We sincerely thank the efforts of the Authors/Contributors from following works: 

* [https://github.com/open-mmlab/mmtracking](https://github.com/open-mmlab/mmtracking)
* [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
* [https://github.com/YuHengsss/YOLOV](https://github.com/YuHengsss/YOLOV)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection](https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection)
</details>
