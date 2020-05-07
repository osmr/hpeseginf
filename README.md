# Simultaneous human pose estimation and segmentation 
Inference Python script for simultaneous human pose estimation and segmentation MXNet models.

# Installation

```
pip3 install -r requirements.txt
```

Download model files form the repo releases.


# Usage

```
python3 demo_mx_hpeseg.py --in-video=<path_to_your_video> --out-video=<path_to_resulting_video> --model=<path_to_model_stem> 
python3 demo_tlite_hpeseg.py --in-video=<path_to_your_video> --out-video=<path_to_resulting_video> --model=<path_to_tfl_model>
python3 demo_onnx_hpeseg.py --in-video=<path_to_your_video> --out-video=<path_to_resulting_video> --model=<path_to_onnx_model>
```

Example:

```
python3 demo_mx_hpeseg.py --in-video=../hpeseginf_data/vid2.mp4 --out-video=../hpeseginf_data/hpeseg_result.mp4 --model=../hpeseginf_data/ducnet_mobilenet_w1_coco 
```
