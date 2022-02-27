from argparse import ArgumentParser
from src.utils.img_plot import putCnText
from typing import List
import cv2
import requests
import os
import shutil
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--det_model', type=str, default='dbnet', help='Torch-serve model name')
    parser.add_argument('--recog_model', type=str, default='sar_cn', help='Torch-serve model name')
    parser.add_argument('--inference-addr', type=str,
                        default='127.0.0.1:8080', help='Torch-serve model name')
    parser.add_argument('--slice_height', type=int, default=640, help='Slice inference')
    parser.add_argument('--slice_width', type=int, default=640, help='Slice inference')
    parser.add_argument('--slice_height_overlap', type=float, default=0.0, help='Slice inference')
    parser.add_argument('--slice_width_overlap', type=float, default=0.0, help='Slice inference')
    parser.add_argument('--threshold_det', type=float, default=0.3, help='Threshold of text detection')
    parser.add_argument('--threshold_recog', type=float, default=0.3, help='Threshold of text recognition')
    parser.add_argument('--input_folder', type=str, default='data/', help='Input folder')
    parser.add_argument('--output-folder', type=str, default='output/', help='Output folder')
    args = parser.parse_args()

    url_det = 'http://' + args.inference_addr + '/predictions/' + args.det_model
    url_recog = 'http://' + args.inference_addr + '/predictions/' + args.recog_model

    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)
    os.mkdir(args.output_folder)

    names = os.listdir(args.input_folder)
    names.sort()
    for i, name in enumerate(names):
        img_path = os.path.join(args.input_folder, name)
        img_path_out = os.path.join(args.output_folder, name)
        print(f"\n{'*' * 10} {i + 1}/{len(names)}, {img_path} {'*' * 10}")

        try:

            img = cv2.imread(img_path)

            windows = get_slice_windows(input_h=img.shape[0], input_w=img.shape[1],
                                        sli_h=args.slice_height, sli_w=args.slice_width,
                                        olp_h_ratio=args.slice_height_overlap, olp_w_ratio=args.slice_width_overlap)
            for j, win in enumerate(windows):
                print(f"windows: {j + 1}/{len(windows)}")

                data = cv2.imencode('.jpg', img[win[1]:win[3], win[0]:win[2], :])[1].tobytes()
                response = requests.post(url_det, data)
                boundary_result = response.json()['boundary_result']
                print(boundary_result)

                for poly in boundary_result:
                    score_det = poly[-1]
                    if score_det < args.threshold_det:
                        continue
                    poly = [int(v) for v in poly]
                    poly = [int(poly[k]) + win[0] if k % 2 == 0 else int(poly[k]) + win[1] for k in range(len(poly)-1)]
                    xx = poly[0::2]
                    yy = poly[1::2]
                    data_lp = cv2.imencode('.jpg', img[min(yy):max(yy), min(xx):max(xx), :])[1].tobytes()
                    response = requests.post(url_recog, data_lp)
                    print(response.json())
                    text, score_ocr = response.json()['text'], response.json()['score']
                    if score_ocr > args.threshold_recog:
                        cv2.polylines(img, pts=np.array([poly[:8]]).reshape(1, 4, 2),
                                      color=(0, 0, 255),
                                      isClosed=True,
                                      thickness=5)
                        img = putCnText(img,
                                        text=f'{text} | {score_det:.2f} | {score_ocr:.2f}',
                                        org=(min(xx), min(yy) - 40),
                                        font='font/simsun.ttc',
                                        textColor=(255, 0, 0),
                                        textSize=40)

            cv2.imwrite(img_path_out, img)
            print(f"{img_path_out}")
        except Exception as e:
            print('!!! error !!!')
            print(e)

    print('Done.')
