"""
    Script for demonstration of HPE-Seg task solving on a video (via TFLite).
"""

import os
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf


def parse_args():
    """
    Create python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="HPE-Seg on a video (via TFLite)",
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
        "--model",
        type=str,
        required=True,
        help="path to TFLite model")
    parser.add_argument(
        "--use-gpus",
        type=int,
        default=1,
        help="use gpus (0/1)")
    parser.add_argument(
        "--keypoints",
        type=int,
        default=17,
        help="number of HPE-joints. options are 17 for COOO and 16 for MHPv2")
    parser.add_argument(
        "--mean-rgb",
        nargs=3,
        type=float,
        default=(0.485, 0.456, 0.406),
        help="mean of RGB channels in the video")
    parser.add_argument(
        "--std-rgb",
        nargs=3,
        type=float,
        default=(0.229, 0.224, 0.225),
        help="STD of RGB channels in the video")

    args = parser.parse_args()
    return args


def _blur_image(image):
    """
    Blur the input video to remove noise.

    Parameters
    ----------
    image : np.ndarray
        Image.

    Returns
    -------
    np.ndarray
        Resulting image.
    """
    image = cv2.medianBlur(image, 5)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def _scale_image_linear(image, target_size):
    """
    Resize image with keeping aspect ratio.

    Parameters
    ----------
    image : np.ndarray
        Image.
    target_size : (int, int)
        Size.

    Returns
    -------
    np.ndarray
        Resulting image.
    """
    h, w = image.shape[:2]
    oh = target_size[0]
    ow = int(float(w * oh) / h)
    image = cv2.resize(image, dsize=(ow, oh), interpolation=cv2.INTER_LINEAR)
    scale_factor = float(oh) / h
    return image, scale_factor


def _crop_image_centaral(image, target_size):
    """
    Central crop of the image.

    Parameters
    ----------
    image : np.ndarray
        Image.
    target_size : (int, int)
        Size.

    Returns
    -------
    np.ndarray
        Resulting image.
    """
    h, w = image.shape[:2]
    th, tw = target_size
    ih = int(round(0.5 * (h - th)))
    jw = int(round(0.5 * (w - tw)))
    image = image[ih:(ih + th), jw:(jw + tw), :]
    shift_value = (jw, ih)
    return image, shift_value


def _convert_to_tensor(image,
                       mean_rgb,
                       std_rgb):
    """
    Convert an image to MXNet tensor.

    Parameters
    ----------
    image : np.ndarray
        Image.
    mean_rgb : tuple of 3 float
        Mean of RGB values.
    std_rgb : tuple of 3 float
        STD of RGB values.

    Returns
    -------
    mx.nd.NDArray
        Resulting image.
    """
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)

    x = image.astype(np.float32)
    x = x / 255.0
    x = (x - np.array(mean_rgb)) / np.array(std_rgb)

    x = np.expand_dims(x, axis=0)
    return x


def _smooth_mask_edges(src_mask):
    """
    Smooth the mask after upscale.

    Parameters
    ----------
    src_mask : np.ndarray
        Mask.

    Returns
    -------
    np.ndarray
        Smoothed mask.
    """
    dst_mask = src_mask
    dst_mask = cv2.blur(dst_mask, (5, 5))
    dst_mask[dst_mask > 127] = 255
    dst_mask[dst_mask <= 127] = 0
    return dst_mask


def _expand_mask_central(src_mask, target_size):
    """
    Central expand the mask.

    Parameters
    ----------
    src_mask : np.ndarray
        Mask.
    target_size : (int, int)
        Size.

    Returns
    -------
    np.ndarray
        Expanded mask.
    """
    dst_mask = np.zeros(shape=target_size, dtype=np.uint8)

    h, w = src_mask.shape
    ch, cw = target_size
    x1 = int(round(0.5 * (cw - w)))
    y1 = int(round(0.5 * (ch - h)))
    dst_mask[y1:(y1 + h), x1:(x1 + w)] = src_mask

    return dst_mask


def _draw_mask_on_image(src_image, mask):
    """
    Draw a mask on an image.

    Parameters
    ----------
    src_image : np.ndarray
        Image.
    mask : np.ndarray
        Mask.

    Returns
    -------
    np.ndarray
        Image with mask.
    """
    dst_image = src_image.copy()
    dst_image_g = dst_image[:, :, 1]
    dst_image_g[mask <= 127] = 255
    dst_image_b = dst_image[:, :, 0]
    dst_image_b[mask > 127] = 255
    return dst_image


def _calc_pts_from_heatmap(heatmap):
    """
    Calculate points and scores from heatmaps.

    Parameters
    ----------
    heatmap : mx.nd.NDArray
        Heatmaps with batch.

    Returns
    -------
    pts : mx.nd.NDArray
        Points & scores.
    """
    heatmap = heatmap.transpose(0, 3, 1, 2)
    vector_dim = 2
    batch = heatmap.shape[0]
    channels = heatmap.shape[1]
    in_size = heatmap.shape[2:]
    heatmap_vector = heatmap.reshape((batch, channels, -1))
    indices = np.expand_dims(heatmap_vector.argmax(axis=vector_dim), axis=vector_dim).astype(np.float32)
    scores = np.max(heatmap_vector, axis=vector_dim, keepdims=True)
    scores_mask = (scores > 0.0).astype(np.float32)
    pts_x = (indices % in_size[1]) * scores_mask
    pts_y = (indices // in_size[1]) * scores_mask
    pts = np.concatenate((pts_x, pts_y, scores), axis=vector_dim)
    return pts


def cv_plot_keypoints(image,
                      pts,
                      scale,
                      shift,
                      keypoint_thresh,
                      num_keypoints):
    """
    Visualize keypoints with OpenCV.

    Parameters
    ----------
    image : np.ndarray or mx.nd.NDArray
        Image with shape `H, W, 3`.
    pts : np.ndarray or mx.nd.NDArray
        Array with shape `Batch, N_Joints, 3`.
    scale : float
        The scale of output image, which may affect the positions of boxes.
    shift : (int, int)
        Shift points on image (w, h).
    keypoint_thresh : float
        Keypoints with confidence less than `keypoint_thresh` will be ignored in display.
    num_keypoints : int
        Number of HPE-joints (17 for COOO and 16 for MHPv2).

    Returns
    -------
    np.ndarray
        The image with estimated pose.
    """
    import matplotlib.pyplot as plt

    if num_keypoints == 16:
        joint_pairs = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [10, 11], [11, 12], [13, 14], [14, 15],
                       [2, 6], [3, 6], [12, 8], [13, 8]]
    else:
        joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
                       [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]]

    for pts_i in pts:
        joint_visible = pts_i[:, 2] > keypoint_thresh
        colormap_index = np.linspace(0, 1, len(joint_pairs))
        pts_i[:, :2] *= scale
        shift = [int(scale * sh) for sh in shift]
        for cm_ind, jp in zip(colormap_index, joint_pairs):
            if joint_visible[jp[0]] and joint_visible[jp[1]]:
                cm_color = tuple([int(x * 255) for x in plt.cm.cool(cm_ind)[:3]])
                pt1 = (int(pts_i[jp[0], 0]) + shift[0], int(pts_i[jp[0], 1]) + shift[1])
                pt2 = (int(pts_i[jp[1], 0]) + shift[0], int(pts_i[jp[1], 1]) + shift[1])
                cv2.line(image, pt1, pt2, cm_color, 3)

    # cv2.imshow("image", image)
    # cv2.waitKey(10)

    return image


def main():
    """
    Main body of script.
    """
    args = parse_args()
    in_video = os.path.expanduser(args.in_video)
    if not os.path.exists(in_video):
        raise Exception("Input video doesn't exist: {}".format(in_video))
    assert os.path.isfile(in_video)

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    total_time = 0

    cap = cv2.VideoCapture(in_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720), True)

    max_num_frames = 100000
    for i in range(max_num_frames):
        ret, frame = cap.read()
        if frame is None:
            break

        if frame.shape[:2] == (720, 1280):
            frame_resized = frame.copy()
        else:
            frame_resized = cv2.resize(frame, dsize=(1280, 720), interpolation=cv2.INTER_LINEAR)

        image0 = _blur_image(frame_resized)
        image1, scale_factor1 = _scale_image_linear(image0, target_size=(192, 256))
        image2, shift_value = _crop_image_centaral(image1, target_size=(192, 256))
        image3 = _convert_to_tensor(
            image2,
            mean_rgb=args.mean_rgb,
            std_rgb=args.std_rgb)

        tic = time.time()

        interpreter.set_tensor(input_details[0]["index"], image3.astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        mask0 = (output[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        pts0 = _calc_pts_from_heatmap(output[:, :, :, 1:])
        total_time += (time.time() - tic)

        mask1, _ = _scale_image_linear(mask0, target_size=(720, 1280))
        mask2 = _smooth_mask_edges(mask1)
        mask3 = _expand_mask_central(mask2, target_size=(720, 1280))
        res_image0 = _draw_mask_on_image(src_image=frame_resized, mask=mask3)
        res_image1 = cv_plot_keypoints(
            image=res_image0,
            pts=pts0,
            scale=(1.0 / scale_factor1),
            shift=shift_value,
            keypoint_thresh=0.3,
            num_keypoints=args.keypoints)

        print(i)

        if video_writer:
            video_writer.write(res_image1.copy())

    cap.release()
    video_writer.release()

    print("Time cost: {:.4f} sec".format(total_time / (i + 1)))


if __name__ == "__main__":
    main()
