from src.version import __version__
from src.ocr_client import OCRClient
from loguru import logger
from argparse import ArgumentParser, Namespace
import cv2
import os
import shutil
import sys
import yaml


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='YAML config file, e.g. config/config.yaml')
    opt = parser.parse_args()

    try:
        # read config file
        with open(opt.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        args = Namespace(**cfg)
        args_log = Namespace(**args.log)
        args_io = Namespace(**args.io)
    except Exception as e:
        logger.error(e)
        sys.exit(-1)

    # logger config
    logger.add(sink=args_log.sink,
               rotation=args_log.rotation,
               retention=args_log.retention,
               compression=args_log.compression,
               enqueue=True)

    logger.info(f"Program version: {__version__}")
    logger.info(f"Program config:\n{yaml.safe_dump(cfg, sort_keys=False)}")

    # ocr client
    try:
        ocr_client = OCRClient(**args.ts)
    except Exception as e:
        logger.error(e)
        sys.exit(-1)

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

            # send OCR request
            img = cv2.imread(img_path)
            result, img_draw = ocr_client.read_text(img, draw=True, **args.infer)

            # print result
            result_str = "[\n\t" + "\n\t".join([f"{elem}," for elem in result]) + "\n]"
            logger.info(f"recog result:\n{result_str}")

            # save image
            img_path_out = os.path.join(args_io.output_folder, name)
            cv2.imwrite(img_path_out, img_draw)
            logger.info(f"save image to {img_path_out}")
        except Exception as e:
            logger.error(e)

    logger.info('Done.')
