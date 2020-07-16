"""
    Script for demonstration of face masking.
"""

import random
import os
import argparse
import time
import json
import cv2
import numpy as np


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
        "--mask-image",
        type=str,
        required=True,
        help="path to mask image (with transparency)")
    parser.add_argument(
        "--mask-meta",
        type=str,
        required=True,
        help="path to mask meta information")
    parser.add_argument(
        "--foreground",
        type=str,
        required=True,
        help="path to foreground image (with transparency)")
    parser.add_argument(
        "--face-cascade",
        type=str,
        required=True,
        help="path to face detection cascade")
    parser.add_argument(
        "--mark-cascade",
        type=str,
        required=True,
        help="path to face landmark detection cascade")
    parser.add_argument(
        "--resize",
        type=float,
        default=0.5,
        help="Resize scale factor for face detection")

    args = parser.parse_args()
    return args


def get_mask(mask_image_path):
    """
    Get mask.

    Parameters
    ----------
    mask_image_path : str
        Path to mask image.

    Returns
    -------
    mask : np.ndarray
        Mask (BGR).
    mask_alpha : np.ndarray
        Mask alpha channel (gray).
    width : int
        Mask width.
    aspect : float
        Mask aspect ratio.
    """
    mask_image_src = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
    mask = mask_image_src[:, :, :3]
    mask_alpha = mask_image_src[:, :, 3]
    mask_alpha[mask_alpha >= 127] = 255
    mask_alpha[mask_alpha < 127] = 0
    height, width = mask.shape[:2]
    aspect = float(height / width)
    return mask, mask_alpha, width, aspect


def get_mask_pts(mask_meta_path):
    """
    Get mask points.

    Parameters
    ----------
    mask_meta_path : str
        Path to mask meta information.

    Returns
    -------
    mask_pts : np.ndarray
        Mask key points.
    """
    with open(mask_meta_path) as json_file:
        mask_meta = json.load(json_file)
    mask_pts = [
        [mask_meta["right_eye"]["x"], mask_meta["right_eye"]["y"]],
        [mask_meta["left_eye"]["x"], mask_meta["left_eye"]["y"]],
        [mask_meta["right_mouth"]["x"], mask_meta["right_mouth"]["y"]],
        [mask_meta["left_mouth"]["x"], mask_meta["left_mouth"]["y"]]
    ]
    mask_pts = np.array(mask_pts, np.float32)
    return mask_pts


def get_foreground(foreground_path, frame_width, frame_height):
    """
    Get foreground.

    Parameters
    ----------
    foreground_path : str
        Path to foreground image.
    frame_width : int
        Frame width.
    frame_height : int
        Frame height.

    Returns
    -------
    fg : np.ndarray
        Foreground (BGR).
    fg_alpha : np.ndarray
        Foreground alpha channel (gray).
    """
    fg_image_src = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
    fg = fg_image_src[:, :, :3]
    fg_alpha = fg_image_src[:, :, 3]
    fg_alpha[fg_alpha >= 127] = 255
    fg_alpha[fg_alpha < 127] = 0
    fg_height, fg_width = fg.shape[:2]
    if (fg_height != frame_height) or (fg_width != frame_width):
        fg = cv2.resize(fg, dsize=(frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
        fg_alpha = cv2.resize(fg_alpha, dsize=(frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
    return fg, fg_alpha


def main():
    """
    Main body of script.
    """
    args = parse_args()
    in_video = os.path.expanduser(args.in_video)
    if not os.path.exists(in_video):
        raise Exception("Input video doesn't exist: {}".format(in_video))
    assert os.path.isfile(in_video)

    frame_scale = args.resize
    cascade = cv2.CascadeClassifier(args.face_cascade)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(args.mark_cascade)

    cap = cv2.VideoCapture(in_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    mask, mask_alpha, mw, ma = get_mask(mask_image_path=args.mask_image)
    mask_pts = get_mask_pts(mask_meta_path=args.mask_meta)
    fg, fg_alpha = get_foreground(
        foreground_path=args.foreground,
        frame_width=frame_width,
        frame_height=frame_height)

    video_writer = cv2.VideoWriter(
        args.out_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height),
        True)

    max_num_frames = 100000
    tic = time.time()
    for i in range(max_num_frames):
        ret, frame = cap.read()
        if frame is None:
            break

        frame_resized = cv2.resize(frame, dsize=None, fx=frame_scale, fy=frame_scale, interpolation=cv2.INTER_AREA)

        # face_rects = cascade.detectMultiScale(
        #     frame_resized,
        #     scaleFactor=1.1,
        #     minNeighbors=4,
        #     minSize=(100, 100),
        #     maxSize=(200, 200))
        face_rects = cascade.detectMultiScale(
            frame_resized,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(20, 20),
            maxSize=(50, 50))

        frame_fg = frame.copy()
        frame_fg[fg_alpha > 127] = fg[fg_alpha > 127]

        if len(face_rects) != 0:
            ok, landmarks = facemark.fit(frame_resized, faces=face_rects)

            for landmarks_j in landmarks:
                landmarks_j /= frame_scale
                right_eye = 0.5 * (landmarks_j[0, 36] + landmarks_j[0, 39])
                left_eye = 0.5 * (landmarks_j[0, 42] + landmarks_j[0, 45])
                # nose = landmarks_j[0, 30]
                right_mouth = landmarks_j[0, 48]
                left_mouth = landmarks_j[0, 54]
                face_pts = np.stack((right_eye,left_eye, right_mouth, left_mouth))
                # h, status = cv2.findHomography(srcPoints=mask_pts, dstPoints=face_pts)
                h = cv2.getPerspectiveTransform(src=mask_pts, dst=face_pts)
                mask_frame = cv2.warpPerspective(
                    src=mask,
                    M=h,
                    dsize=(frame_width, frame_height),
                    flags=cv2.INTER_LINEAR)
                mask_alpha_frame = cv2.warpPerspective(
                    src=mask_alpha,
                    M=h,
                    dsize=(frame_width, frame_height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0)
                frame_fg[mask_alpha_frame > 127] = mask_frame[mask_alpha_frame > 127]

        cv2.imshow("image_output", frame_fg)
        cv2.waitKey(10)
        print(i)

        if video_writer:
            video_writer.write(frame_fg.copy())

    total_time = (time.time() - tic)

    cap.release()
    video_writer.release()

    print("Time cost: {:.4f} sec".format(total_time / (i + 1)))


if __name__ == "__main__":
    main()
