from argparse import ArgumentParser
from src.utils.img_plot import putCnText
import cv2
import requests
import os
import shutil
import numpy as np


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--det_model', type=str, default='dbnet', help='Torch-serve model name')
    parser.add_argument('--recog_model', type=str, default='sar_cn', help='Torch-serve model name')
    parser.add_argument('--inference-addr', type=str,
                        default='127.0.0.1:8080', help='Torch-serve model name')
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

        img = cv2.imread(img_path)

        data = cv2.imencode('.jpg', img)[1].tobytes()
        response = requests.post(url_det, data)
        boundary_result = response.json()['boundary_result']
        print(boundary_result)

        for poly in boundary_result:
            score_det = poly[-1]
            if score_det < args.threshold_det:
                continue
            poly = [int(v) for v in poly]
            xx = poly[0:8:2]
            yy = poly[1:8:2]
            img_lp = img[min(yy):max(yy), min(xx):max(xx), :]
            data_lp = cv2.imencode('.jpg', img_lp)[1].tobytes()
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

    print('Done.')
