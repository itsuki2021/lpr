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

            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow("image", img)
            cv2.waitKey(2)
    except Exception as e:
        print(f'\n!!! error !!!\n\t{e}')
        sys.exit(-1)

    print("Done.")
