from src.model.hyperlpr import LPR
from src.utils.img_plot import putCnText
from http.client import HTTPConnection
from PIL import Image
from argparse import ArgumentParser
import numpy as np
import json
import hashlib
import time
import cv2
import io
import sys
import yaml


def encrypt(s: str) -> str:
    """ md5加密，小写32位

    :param s: 输入字符串
    :return: 加密字符串
    """
    enc = hashlib.md5()
    enc.update(s.encode('utf-8'))
    return (enc.hexdigest()).lower()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='yaml config file')
    opt = parser.parse_args()

    # 读取参数
    try:
        with open(opt.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f'Can not found {opt.config}\n\t{e}')
        sys.exit(-1)

    # 登录LiveGBS，获取token
    url_login = f'/api/v1/login?' \
                f'username={cfg["usr"]}&password={encrypt(cfg["pwd"])}'
    headers = {'Content-Type': 'text/html;charset=utf-8'}
    try:
        conn = HTTPConnection(host=cfg["host"], port=cfg["port"])
        conn.request(method=cfg["method"], url=url_login, headers=headers)
        rsp = conn.getresponse()
        print('Login')
        print('\tResponse.status' + str(rsp.status))
        print('\tRespond.reason' + str(rsp.reason))
        bts = rsp.read()
        data = json.loads(str(bts, 'utf-8'))
        token = data['URLToken']
    except Exception as e:
        print(f'\nFailed to login\n\t{e}')
        sys.exit(-1)

    # 请求实时快照
    url_snap = f'/api/v1/device/channelsnap?' \
               f'token={token}' \
               f'&serial={cfg["serial"]}' \
               f'&channel={cfg["channel"]}' \
               f'&realtime=true'
    try:
        # 加载识别模型
        PR = LPR(cfg["checkpoint_folder"], conf=cfg["conf"])
        for i in range(cfg["num_test"]):
            tic = time.time()
            conn.request(method=cfg["method"], url=url_snap)
            rsp = conn.getresponse()
            print(f'\nSnap {i}')
            print('\tResponse.status' + str(rsp.status))
            print('\tRespond.reason' + str(rsp.reason))
            bts = rsp.read()
            img = np.asarray(Image.open(io.BytesIO(bts)))
            conn.close()

            # 推理
            print(f'Inference {i}')
            result = PR.plate_recognition(img, minSize=30, charSelectionDeskew=True)
            toc = time.time()
            print(f'\t{result}')
            print(f'Total cost\n\t{(toc - tic) * 1000:.2f}ms')
            for i in range(len(result)):
                txt, conf, bbox = result[i]     # 字符，自信度，边界框
                cv2.rectangle(img, pt1=bbox[:2], pt2=bbox[2:], color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                img = putCnText(img,
                                text=f'{txt} | {conf:.2f}',
                                org=(bbox[0], bbox[1] - 30),
                                font='font/simsun.ttc',
                                textColor=(255, 0, 0),
                                textSize=30)
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow("image", img)
            cv2.waitKey(2)
    except Exception as e:
        print(f'\nError in inference\n\t{e}')
        sys.exit(-1)

    print("Done.")
