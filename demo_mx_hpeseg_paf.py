"""
    Script for demonstration of multi-pose HPE-Seg task solving on a video (via pure MXNet).
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
        description="Multi-pose HPE-Seg on a video (via pure MXNet)",
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
        help="path to MXNet model stem (path to common part of json/params files)")
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
    parser.add_argument(
        "--use-paf",
        type=int,
        default=1,
        help="use PAF for multi HPE (0/1)")

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
                       mean_rgb=(0.485, 0.456, 0.406),
                       std_rgb=(0.229, 0.224, 0.225),
                       ctx=mx.cpu()):
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
    ctx : mx.Context
        The context of the tensor.

    Returns
    -------
    mx.nd.NDArray
        Resulting image.
    """
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)

    x = image.astype(np.float32)
    x = x / 255.0
    x = (x - np.array(mean_rgb)) / np.array(std_rgb)

    x = x.transpose(2, 0, 1)
    x = np.expand_dims(x, axis=0)
    x = mx.nd.array(x, ctx=ctx)
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


def _calc_pts_from_heatmap(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = mx.nd.argmax(heatmaps_reshaped, 2)
    maxvals = mx.nd.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = mx.nd.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = mx.nd.floor((preds[:, :, 1]) / width)

    pred_mask = mx.nd.tile(mx.nd.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def extract_keypoints(heatmap,
                      all_keypoints,
                      total_keypoint_num,
                      heatmap_thr=0.33,
                      peak_radius=24.0):
    heatmap[heatmap < heatmap_thr] = 0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode="constant")
    heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 1:heatmap_with_borders.shape[1]-1]
    heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 2:heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 0:heatmap_with_borders.shape[1]-2]
    heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1]-1]
    heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0]-2, 1:heatmap_with_borders.shape[1]-1]

    heatmap_peaks = (heatmap_center > heatmap_left) &\
                    (heatmap_center > heatmap_right) &\
                    (heatmap_center > heatmap_up) &\
                    (heatmap_center > heatmap_down)
    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]

    peaks_idx = np.nonzero(heatmap_peaks)
    keypoints = np.vstack(peaks_idx)
    sort_idx = heatmap[peaks_idx].argsort()[::-1]
    keypoints = keypoints[:, sort_idx]

    suppressed = np.zeros(len(sort_idx), np.uint8)
    keypoints_with_score_and_id = []
    keypoint_num = 0
    for i in range(len(sort_idx)):
        if suppressed[i]:
            continue
        for j in range(i+1, len(sort_idx)):
            if np.math.sqrt(
                    (keypoints[0][i] - keypoints[0][j]) ** 2 + (keypoints[1][i] - keypoints[1][j]) ** 2) < peak_radius:
                suppressed[j] = 1
        keypoint_with_score_and_id = (
            keypoints[1][i],
            keypoints[0][i],
            heatmap[keypoints[0][i], keypoints[1][i]],
            total_keypoint_num + keypoint_num)
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)
        keypoint_num += 1
    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num


def linspace2d(start, stop, n=10):
    points = 1 / (n - 1) * (stop - start)
    return points[:, None] * np.arange(n) + start[:, None]


def convert_poses_to_pts_format(pose_entries,
                                all_keypoints):
    pts = []
    scores = []
    for pose_entry_i in pose_entries:
        if len(pose_entry_i) == 0:
            continue
        pts_i = np.full((18, 2), np.nan, np.float32)
        scores_i = np.full((18,), np.nan, np.float32)
        for j, keypoint_id in enumerate(pose_entry_i[:-2]):
            if keypoint_id != -1:
                pts_i[j] = all_keypoints[int(keypoint_id), 0:2]
                scores_i[j] = all_keypoints[int(keypoint_id), 2]
        pts.append(pts_i)
        scores.append(scores_i)
    return np.array(pts), np.array(scores)


def group_keypoints(all_keypoints_by_type,
                    pafs,
                    pose_entry_size=20,
                    min_paf_dot_score=0.02,
                    min_paf_suc_score=0.5):
    from operator import itemgetter

    BODY_PARTS_KPT_IDS = [
        [6, 17], [8, 6], [10, 8],
        [5, 17], [7, 5], [9, 7],
        [12, 17], [14, 12], [16, 14],
        [11, 17], [13, 11], [15, 13],
        [0, 17],
        [2, 0], [4, 2],
        [1, 0], [3, 1],
    ]
    BODY_PARTS_PAF_IDS = (
        [4, 5], [2, 3], [0, 1],
        [10, 11], [8, 9], [6, 7],
        [16, 17], [14, 15], [12, 13],
        [22, 23], [20, 21], [18, 19],
        [32, 33],
        [26, 27], [24, 25],
        [30, 31], [28, 29],
    )

    pose_entries = []
    all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])
    for part_id in range(len(pafs) // 2):
        part_pafs_id = BODY_PARTS_PAF_IDS[part_id]
        part_pafs = pafs[part_pafs_id]
        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
        kpts_a = all_keypoints_by_type[kpt_a_id]
        kpts_b = all_keypoints_by_type[kpt_b_id]
        num_kpts_a = len(kpts_a)
        num_kpts_b = len(kpts_b)

        if num_kpts_a == 0 and num_kpts_b == 0:  # no keypoints for such body part
            continue
        elif num_kpts_a == 0:  # body part has just 'b' keypoints
            for i in range(num_kpts_b):
                num = 0
                for j in range(len(pose_entries)):  # check if already in some pose, was added by another body part
                    if pose_entries[j][kpt_b_id] == kpts_b[i][3]:
                        num += 1
                        continue
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_b_id] = kpts_b[i][3]  # keypoint idx
                    pose_entry[-1] = 1  # num keypoints in pose
                    pose_entry[-2] = kpts_b[i][2]  # pose score
                    pose_entries.append(pose_entry)
            continue
        elif num_kpts_b == 0:  # body part has just 'a' keypoints
            for i in range(num_kpts_a):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == kpts_a[i][3]:
                        num += 1
                        continue
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = kpts_a[i][3]
                    pose_entry[-1] = 1
                    pose_entry[-2] = kpts_a[i][2]
                    pose_entries.append(pose_entry)
            continue

        connections = []
        for i in range(num_kpts_a):
            kpt_a = np.array(kpts_a[i][0:2])
            for j in range(num_kpts_b):
                kpt_b = np.array(kpts_b[j][0:2])
                mid_point = [(), ()]
                mid_point[0] = (int(round((kpt_a[0] + kpt_b[0]) * 0.5)),
                                int(round((kpt_a[1] + kpt_b[1]) * 0.5)))
                mid_point[1] = mid_point[0]

                vec = [kpt_b[0] - kpt_a[0], kpt_b[1] - kpt_a[1]]
                vec_norm = np.math.sqrt(vec[0] ** 2 + vec[1] ** 2)
                if vec_norm == 0:
                    continue
                vec[0] /= vec_norm
                vec[1] /= vec_norm
                cur_point_score = (vec[0] * part_pafs[0, mid_point[0][1], mid_point[0][0]] +
                                   vec[1] * part_pafs[1, mid_point[1][1], mid_point[1][0]])

                height_n = pafs.shape[1] // 2
                success_ratio = 0
                point_num = 10  # number of points to integration over paf
                if cur_point_score > -100:
                    passed_point_score = 0
                    passed_point_num = 0
                    x, y = linspace2d(kpt_a, kpt_b)
                    for point_idx in range(point_num):
                        px = int(round(x[point_idx]))
                        py = int(round(y[point_idx]))
                        paf = part_pafs[0:2, py, px]
                        cur_point_score = vec[0] * paf[0] + vec[1] * paf[1]
                        if cur_point_score > min_paf_dot_score:
                            passed_point_score += cur_point_score
                            passed_point_num += 1
                    success_ratio = passed_point_num / point_num
                    ratio = 0
                    if passed_point_num > 0:
                        ratio = passed_point_score / passed_point_num
                    ratio += min(height_n / vec_norm - 1, 0)
                if ratio > 0 and success_ratio >= min_paf_suc_score:
                    score_all = ratio + kpts_a[i][2] + kpts_b[j][2]
                    connections.append([i, j, ratio, score_all])
        if len(connections) > 1:
            connections = sorted(connections, key=itemgetter(3), reverse=True)

        num_connections = min(num_kpts_a, num_kpts_b)
        has_kpt_a = np.zeros(num_kpts_a, dtype=np.int32)
        has_kpt_b = np.zeros(num_kpts_b, dtype=np.int32)
        filtered_connections = []
        for row in range(len(connections)):
            if len(filtered_connections) == num_connections:
                break
            i, j, cur_point_score = connections[row][0:3]
            if not has_kpt_a[i] and not has_kpt_b[j]:
                filtered_connections.append([kpts_a[i][3], kpts_b[j][3], cur_point_score])
                has_kpt_a[i] = 1
                has_kpt_b[j] = 1
        connections = filtered_connections
        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]
            for i in range(len(connections)):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
        else:
            for i in range(len(connections)):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_b_id] == connections[i][1]:
                        pose_entries[j][kpt_a_id] = connections[i][0]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += all_keypoints[connections[i][0], 2] + connections[i][2]
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                    pose_entries.append(pose_entry)

    filtered_entries = []
    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
            continue
        filtered_entries.append(pose_entries[i])
    pose_entries = np.asarray(filtered_entries)
    pts, scores = convert_poses_to_pts_format(
        pose_entries=pose_entries,
        all_keypoints=all_keypoints)
    return pts, scores


def _calc_pts_from_paf(output):
    outputs_np = output[0].asnumpy()
    heatmap = outputs_np[1:19]
    paf = outputs_np[19:]

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(18):
        total_keypoints_num += extract_keypoints(
            heatmap=heatmap[kpt_idx],
            all_keypoints=all_keypoints_by_type,
            total_keypoint_num=total_keypoints_num)
    pts, scores = group_keypoints(all_keypoints_by_type, paf)
    return pts, scores


def cv_plot_frame_num(image,
                      frame_num):
    """
    Visualize frame number with OpenCV.
    """
    cv2.putText(image, "{}".format(frame_num), org=(5, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 0, 255), thickness=1)


def cv_plot_keypoints(image,
                      pts,
                      scores,
                      scale=1.0,
                      shift=(0, 0),
                      keypoint_thresh=0.3):
    """
    Visualize keypoints with OpenCV.

    Parameters
    ----------
    image : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    pts : numpy.ndarray or mxnet.nd.NDArray
        Array with shape `Batch, N_Joints, 2`.
    scale : float
        The scale of output image, which may affect the positions of boxes
    shift : (int, int)
        Shift points on image (w, h).
    keypoint_thresh : float, optional, default 0.2
        Keypoints with confidence less than `keypoint_thresh` will be ignored in display.

    Returns
    -------
    numpy.ndarray
        The image with estimated pose.
    """
    import matplotlib.pyplot as plt

    joint_pairs = [
        [10, 8], [8, 6], [6, 17],
        [9, 7], [7, 5], [5, 17],
        [16, 14], [14, 12], [12, 17],
        [15, 13], [13, 11], [11, 17],
        [4, 2], [2, 0],
        [3, 1], [1, 0],
        [0, 17],
    ]

    shift = [int(scale * sh) for sh in shift]
    for i, (pts_i, scores_i) in enumerate(zip(pts, scores)):
        pts_ii = pts_i.copy()
        joint_visible = scores_i[:] > keypoint_thresh
        colormap_index = np.linspace(0, 1, len(joint_pairs))
        pts_ii *= scale
        for cm_ind, jp in zip(colormap_index, joint_pairs):
            if joint_visible[jp[0]] and joint_visible[jp[1]]:
                cm_color = tuple([int(x * 255) for x in plt.cm.cool(cm_ind)[:3]])
                pt1 = (int(pts_ii[jp[0], 0]) + shift[0], int(pts_ii[jp[0], 1]) + shift[1])
                pt2 = (int(pts_ii[jp[1], 0]) + shift[0], int(pts_ii[jp[1], 1]) + shift[1])
                cv2.line(image, pt1, pt2, cm_color, 3)

    cv2.imshow("image", image)
    cv2.waitKey(10)

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

    ctx = mx.gpu(0) if args.use_gpus == 1 else mx.cpu()

    sym_file_path = args.model + "-symbol.json"
    params_file_path = args.model + "-0000.params"
    net = mx.gluon.SymbolBlock(
        outputs=mx.sym.load(sym_file_path),
        inputs=mx.sym.var("data", dtype=np.float32))
    net.collect_params().load(params_file_path, ctx=ctx)

    total_time = 0

    w = 1280
    h = 720

    cap = cv2.VideoCapture(in_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), True)

    max_num_frames = 100000
    for i in range(max_num_frames):
        ret, frame = cap.read()
        if frame is None:
            break

        if frame.shape[:2] == (h, w):
            frame_resized = frame.copy()
        else:
            frame_resized = cv2.resize(frame, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

        image0 = _blur_image(frame_resized)
        image1, scale_factor1 = _scale_image_linear(image0, target_size=(192, 256))
        image2, shift_value = _crop_image_centaral(image1, target_size=(192, 256))
        image3 = _convert_to_tensor(
            image2,
            mean_rgb=args.mean_rgb,
            std_rgb=args.std_rgb,
            ctx=ctx)

        tic = time.time()
        net_output = net(image3)
        total_time += (time.time() - tic)

        mask0 = (net_output[0, 0] > 0.5).asnumpy().astype(np.uint8) * 255
        if args.use_paf:
            pts0, scores0 = _calc_pts_from_paf(net_output)
        else:
            pts0, scores0 = _calc_pts_from_heatmap(net_output[:, 1:(1 + 18)])
            pts0 = pts0.asnumpy()
            scores0 = scores0.asnumpy()

        mask1, _ = _scale_image_linear(mask0, target_size=(h, w))
        mask2 = _smooth_mask_edges(mask1)
        mask3 = _expand_mask_central(mask2, target_size=(h, w))
        res_image0 = _draw_mask_on_image(src_image=frame_resized, mask=mask3)
        cv_plot_frame_num(image=res_image0, frame_num=i)
        res_image1 = cv_plot_keypoints(
            image=res_image0,
            pts=pts0,
            scores=scores0,
            scale=(1.0 / scale_factor1),
            shift=shift_value)

        print(i)

        if video_writer:
            video_writer.write(res_image1.copy())

    cap.release()
    video_writer.release()

    print("Time cost: {:.4f} sec".format(total_time / (i + 1)))


if __name__ == "__main__":
    main()
