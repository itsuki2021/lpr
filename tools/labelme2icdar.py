import json
import random
import os
import shutil
from argparse import ArgumentParser
from loguru import logger
from tqdm import tqdm

# license plate number, 1 province + 1 alphabet + N ads
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def make_dirs(path: str, force=False):
    if force:
        shutil.rmtree(path) if os.path.exists(path) else None
        os.makedirs(path)
    else:
        os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root_in', default='../data/labelme_folder/', help='path of labelme')
    parser.add_argument('--root_out', default='../data/chinese_lp_det/', help='path of IcdarDataset')
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
    for i, names in enumerate([json_train, json_test]):
        stage = stages[i]
        img_out_prefix = os.path.join(args.root_out, f'imgs/{stage}')
        ann_out_prefix = os.path.join(args.root_out, f'annotations/{stage}')
        make_dirs(img_out_prefix)
        make_dirs(ann_out_prefix)

        # read json files
        for name in tqdm(names):
            try:
                json_path = os.path.join(args.root_in, name)
                with open(json_path, mode='r', encoding='utf-8') as f:
                    ann = json.load(f)

                ann_text = ""
                img_name = ann['imagePath']
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
                    ann_text += f"{xmin},{ymin},{xmax},{ymin},{xmax},{ymax},{xmin},{ymax},{label}\n"

                if ann_text != "":
                    img_in_path = os.path.join(args.root_in, img_name)
                    img_out_path = os.path.join(img_out_prefix, img_name)
                    shutil.copy(img_in_path, img_out_path)
                    ann_out_path = os.path.join(ann_out_prefix, "gt_" + name.replace(".json", ".txt"))
                    with open(ann_out_path, mode='w+', encoding='utf-8') as f:
                        f.write(ann_text)

            except Exception as e:
                logger.error(name)
                logger.error(e)

    print("Done.")
