import numpy as np


def upscale_detection(detection, max_coords):
    bh, bw = detection[3] - detection[1], detection[2] - detection[0]
    min_x1, min_y1, max_x2, max_y2 = max_coords
    ratio_y1, ratio_y2 = 0.95 + bh / max_y2 / 2, 0.65 + bh / max_y2 / 2
    ratio_x = 0.99 + bw / max_x2 / 2  
    cy, cx = detection[1] + int(bh / 2), detection[0] + int(bw / 2)
    y1, x1 = max(min_y1, cy - int(bh * ratio_y1)), max(min_x1, cx - int(bw * ratio_x))
    y2, x2 = min(max_y2, cy + int(bh * ratio_y2)), min(max_x2, cx + int(bw * ratio_x))
    area = (y2 - y1) * (x2 - x1)
    return [x1, y1, x2, y2, area]

def upscale_detections(detections, coords):
    upscaled_detections = []
    for det in detections: 
        upscaled_detections.append(upscale_detection(det, coords))
    return upscaled_detections



def compute_increased_bbox(bbox, frame_shape, increase_area):
        left, top, right, bot = bbox
        width = right - left
        height = bot - top
    
        left = int(left - increase_area * width)
        top = int(top - increase_area * height)
        right = int(right + increase_area * width)
        bot = int(bot + increase_area * height)

        left = np.clip(left, 0, frame_shape[1])
        right = np.clip(right, 0, frame_shape[1])
        top = np.clip(top, 0, frame_shape[0])
        bot = np.clip(bot, 0, frame_shape[0])

        return [left, top, right, bot]


def compute_aspect_preserved_bbox(bbox, frame_shape, increase_area):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top
 
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    left = np.clip(left, 0, frame_shape[1])
    right = np.clip(right, 0, frame_shape[1])
    top = np.clip(top, 0, frame_shape[0])
    bot = np.clip(bot, 0, frame_shape[0])

    return (left, top, right, bot)

def scale_bboxes(img1_shape, bboxes, img0_shape, ratio_pad=None):

    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    bboxes[:, [0, 2]] -= pad[0]  # x padding
    bboxes[:, [1, 3]] -= pad[1]  # y padding
    bboxes[:, :4] /= gain

    # clip coords
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, img0_shape[0])
    return bboxes


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0] -= pad[0]  # x padding
    coords[:, 1] -= pad[1]  # y padding
    coords[
        :,
    ] /= gain

    # clip coords
    coords[:, 0] = coords[:, 0].clip(0, img0_shape[1])  # x1, x2
    coords[:, 1] = coords[:, 1].clip(0, img0_shape[0])
    return coords