"""
    Script for demonstration of face masking.
"""

import os
import argparse
import time
import cv2
import numpy as np
import mxnet as mx


def parse_args():
    """
    Create python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Face masking on a video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--in-video",
        type=str,
        required=True,
        help="path to an input video")
    parser.add_argument(
        "--out-video",
        type=str,
        required=True,
        help="path to the output video")
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="path to mask image (with transparency)")
    parser.add_argument(
        "--cascade",
        type=str,
        required=True,
        help="path to Haar cascade")

    args = parser.parse_args()
    return args


def main():
    """
    Main body of script.
    """
    args = parse_args()
    in_video = os.path.expanduser(args.in_video)
    if not os.path.exists(in_video):
        raise Exception("Input video doesn't exist: {}".format(in_video))
    assert os.path.isfile(in_video)

    mask_src = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
    mask = mask_src[:, :, :3]
    mask_apha = mask_src[:, :, 3]
    mask_apha[mask_apha >= 127] = 255
    mask_apha[mask_apha < 127] = 0
    mh, mw = mask.shape[:2]
    ma = float(mh / mw)

    cascade = cv2.CascadeClassifier(args.cascade)

    cap = cv2.VideoCapture(in_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720), True)

    max_num_frames = 100000
    tic = time.time()
    for i in range(max_num_frames):
        ret, frame = cap.read()
        if frame is None:
            break

        frame_resized = cv2.resize(frame, dsize=(640, 360), interpolation=cv2.INTER_AREA)
        rects = cascade.detectMultiScale(
            frame_resized,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(8, 8),
            maxSize=(50, 50))

        if len(rects) != 0:
            rx0, ry0, rw, rh = rects[0]
            rx1 = rx0 + rw - 1
            ry1 = ry0 + rh - 1
            cv2.rectangle(frame_resized, (rx0, ry0), (rx1, ry1), (255, 0, 0), 2)

            rm_scale = float(rw) / mw
            ry0m = int(ry0 + 0.5 * (rh - ma * rw))
            mask_scaled = cv2.resize(mask, dsize=None, fx=rm_scale, fy=rm_scale, interpolation=cv2.INTER_LINEAR)
            mask_apha_scaled = cv2.resize(mask_apha, dsize=None, fx=rm_scale, fy=rm_scale, interpolation=cv2.INTER_NEAREST)
            rmh, rmw = mask_scaled.shape[:2]

            frame_resized[ry0m:(ry0m + rmh), rx0:(rx0 + rmw)][mask_apha_scaled > 127] = mask_scaled[mask_apha_scaled > 127]

        image_output = cv2.resize(frame_resized, dsize=(1280, 720), interpolation=cv2.INTER_LINEAR)

        cv2.imshow("image_output", image_output)
        cv2.waitKey(10)

        print(i)

        if video_writer:
            video_writer.write(image_output.copy())

    total_time = (time.time() - tic)

    cap.release()
    video_writer.release()

    print("Time cost: {:.4f} sec".format(total_time / (i + 1)))


if __name__ == "__main__":
    main()
