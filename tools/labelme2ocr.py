import shutil
from argparse import ArgumentParser
from labelme2icdar import make_dirs
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import os
import random
import json
import cv2

# license plate number, 1 province + 1 alphabet + N ads
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root_in', default='../data/labelme_folder/', help='path of labelme')
    parser.add_argument('--root_out', default='../data/chinese_lp_recog/', help='path of IcdarDataset')
    parser.add_argument('--split_rate', default=0.8, help='split rate of training and test data')
    args = parser.parse_args()

    charset = []
    for char in provinces + alphabets + ads:
        if char not in charset:
            charset.append(char)

    # create output folder
    make_dirs(args.root_out, force=True)

    # list json files
    json_all = [name for name in os.listdir(args.root_in) if name.endswith(".json")]
    random.shuffle(json_all)
    stages = ['train', 'test']
    json_train = json_all[:int(len(json_all) * args.split_rate)]
    json_test = json_all[int(len(json_all) * args.split_rate):]

    # for each stage
    ann_out_prefix = os.path.join(args.root_out, "labels/")
    make_dirs(ann_out_prefix)
    for i, names in enumerate([json_train, json_test]):
        stage = stages[i]
        img_out_prefix = os.path.join(args.root_out, f'imgs/{stage}')
        make_dirs(img_out_prefix)
        ann_text = ""

        # read json files
        for name in tqdm(names):
            try:
                json_path = os.path.join(args.root_in, name)
                with open(json_path, mode='r', encoding='utf-8') as f:
                    ann = json.load(f)

                img_name = ann['imagePath']
                img = cv2.imread(os.path.join(args.root_in, img_name))
                for shape in ann["shapes"]:
                    label = shape["label"]
                    for char in label:
                        assert char in charset
                    assert shape["shape_type"] == "rectangle"
                    points = shape["points"]
                    xmin = int(min(points[0][0], points[1][0]))
                    ymin = int(min(points[0][1], points[1][1]))
                    xmax = int(max(points[0][0], points[1][0]))
                    ymax = int(max(points[0][1], points[1][1]))

                    dt_ms = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
                    img_out_path = os.path.join(img_out_prefix, f"{dt_ms}.jpg")
                    ann_text += f'{dt_ms}.jpg {label}\n'
                    img_block = img[ymin:ymax, xmin:xmax, :]
                    cv2.imwrite(img_out_path, img_block)

            except Exception as e:
                logger.error(name)
                logger.error(e)

            with open(os.path.join(ann_out_prefix, f"{stage}.txt"), mode="w+", encoding="utf-8") as f:
                f.write(ann_text)

    shutil.copy("../torchserve/dict_printed_chinese_alpha_lp.txt", ann_out_prefix)

    logger.info("Done.")
