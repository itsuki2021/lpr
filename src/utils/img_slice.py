from typing import List
import numpy as np


def get_slice_windows(input_h: int, input_w: int, sli_h: int, sli_w: int,
                      olp_h_ratio: float, olp_w_ratio: float) -> List[List[int]]:
    """Slices `image` in crops.
    Corner values of each infer will be generated using the `sli_h`,
    `sli_w`, `olp_h_ratio` and `olp_w_ratio`.

    input_h (int): Height of the original image.
    input_w (int): Width of the original image.
    sli_h (int): Height of each infer.
    sli_w (int): Width of each infer.
    olp_h_ratio (float): Fractional overlap in height of each
        infer (e.g. an overlap of 0.2 for a infer of size 100 yields an
        overlap of 20 pixels).
    olp_w_ratio (float): Fractional overlap in width of each
        infer (e.g. an overlap of 0.2 for a infer of size 100 yields an
        overlap of 20 pixels).

    Returns:
        List[List[int]]: List of 4 corner coordinates for each N slices.
            [
                [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                ...
                [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
            ]
    """
    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(olp_h_ratio * sli_h)
    x_overlap = int(olp_w_ratio * sli_w)
    while y_max - y_overlap < input_h:
        x_min = x_max = 0
        y_max = y_min + sli_h
        while x_max - x_overlap < input_w:
            x_max = x_min + sli_w
            if y_max > input_h or x_max > input_w:
                ymax = min(input_h, y_max)
                xmax = min(input_w, x_max)
                slice_bboxes.append([xmax - sli_w, ymax - sli_h, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap

    # remove duplicated bboxes
    slice_bboxes_unique = []
    for slice_bbox in slice_bboxes:
        if slice_bbox not in slice_bboxes_unique:
            slice_bboxes_unique.append(slice_bbox)
    return slice_bboxes_unique


def is_overlapped(bbox1: np.ndarray, bbox2: np.ndarray) -> bool:
    """ whether bboxes is overlapped
    Args:
        bbox1: input bbox, np.array(5,), xmin, ymin, xmax, ymax, score
        bbox2: input bbox, np.array(5,), xmin, ymin, xmax, ymax, score

    Returns:
        overlapped or not
    """
    x1_min, y1_min, x1_max, y1_max = bbox1[:4]
    x2_min, y2_min, x2_max, y2_max = bbox2[:4]
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    if x_min > x_max or y_min > y_max:
        return False

    return True


def bbox_union(bboxes: np.ndarray):
    """

    Args:
        bboxes: input bboxes, np.array(n, 5) (x_min, y_min, x_max, y_max, score; ...)

    Returns:    merged bbox

    """
    bbox = np.zeros(shape=(1, 5), dtype=float)
    bbox[:, 0] = min(bboxes[:, 0])
    bbox[:, 1] = min(bboxes[:, 1])
    bbox[:, 2] = max(bboxes[:, 2])
    bbox[:, 3] = max(bboxes[:, 3])
    bbox[:, 4] = max(bboxes[:, 4])

    return bbox