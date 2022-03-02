from src.ocr_client import OCRClient
from loguru import logger
from argparse import Namespace
import os
import cv2
import shutil
import json


if __name__ == '__main__':
    # read config file
    cfg = {
        "ts": {
            "det_model": 'dbnet',
            "recog_model": 'sar_cn',
            "inference_addr": '192.168.1.10:8080'
        },
        "infer": {
            "slice_height": 736,
            "slice_width": 1333,
            "slice_height_overlap": 0.1,
            "slice_width_overlap": 0.1,
            "threshold_det": 0.7,
            "threshold_recog": 0.7,
        },
        "io": {
            "input_folder": '../data/',
            "output_folder": '../output/'
        }
    }
    args = Namespace(**cfg)
    args_io = Namespace(**args.io)

    # ocr client
    ocr_client = OCRClient(**args.ts)

    # create output folder
    if os.path.exists(args_io.output_folder):
        shutil.rmtree(args_io.output_folder)
    os.mkdir(args_io.output_folder)

    # list directory
    names = os.listdir(args_io.input_folder)
    names.sort()
    for i, name in enumerate(names):
        try:
            img_path = os.path.join(args_io.input_folder, name)
            logger.info(f"{'*' * 20} {i + 1}/{len(names)}, read image: {img_path} {'*' * 20}")

            img = cv2.imread(img_path)
            result = ocr_client.read_text(img, draw=False, **args.infer)
            shapes = []
            for recog in result:
                shapes.append({
                    "label": recog['textrecog'][0],
                    "points": [
                        [recog['textdet'][0], recog['textdet'][1]],
                        [recog['textdet'][2], recog['textdet'][3]]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                })

            ann_labelme = {
                "version": "5.0.0",
                "flags": {},
                "shapes": shapes,
                "imagePath": name,
                "imageData": None,
                "imageHeight": img.shape[0],
                "imageWidth": img.shape[1]
            }

            img_path_out = os.path.join(args_io.output_folder, name)
            ann_path_out = os.path.join(args_io.output_folder, name.replace(".jpg", ".json"))
            cv2.imwrite(img_path_out, img)
            with open(ann_path_out, "w+", encoding='utf-8') as f:
                json.dump(ann_labelme, f, indent=2)

        except Exception as e:
            logger.error(e)

    logger.info("Done.")
