# Simultaneous human pose estimation and segmentation 
Inference Python script for simultaneous human pose estimation and segmentation MXNet models.

# Installation

```
pip3 install -r requirements.txt
```

# Usage

```
python3 demo_mask.py --in-video=<path_to_your_video> --out-video=<path_to_resulting_video> --mask=<path_to_transparent_mask> --cascade=<path_to_opencv_cascade>
```

Example:

```
python3 demo_mask.py --in-video=../hpeseginf_data/vid1.mp4 --out-video=../hpeseginf_data/mask_result.mp4 --mask=../hpeseginf_data/m4.png --cascade=../hpeseginf_data/haarcascade_frontalface_alt.xml 
```
