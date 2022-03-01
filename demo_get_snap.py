from http.client import HTTPConnection
from PIL import Image
from argparse import ArgumentParser
from loguru import logger
import numpy as np
import json
import hashlib
import time
import cv2
import io
import sys


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
    parser.add_argument('--host', type=str, default='8.141.69.166', help='host address, e.g. 0.0.0.0')
    parser.add_argument('--port', type=int, default=10256, help='host port e.g. 8000')
    parser.add_argument('--usr', type=str, default='admin', help='user name')
    parser.add_argument('--pwd', type=str, default='admin', help='password')
    parser.add_argument('--serial', type=str, default='34020000001110000031', help='LiveGBS device serial id')
    parser.add_argument('--channel', type=int, default=1, help='LiveGBS stream channel')
    parser.add_argument('--num_test', type=int, default=10, help='number of test')
    # 读取参数
    opt = parser.parse_args()

    # 登录LiveGBS，获取token
    url_login = f'/api/v1/login?' \
                f'username={opt.user}&password={encrypt(opt.pwd)}'
    headers = {'Content-Type': 'text/html;charset=utf-8'}
    try:
        conn = HTTPConnection(host=opt.host, port=opt.port)
        conn.request(method="GET", url=url_login, headers=headers)
        rsp = conn.getresponse()
        logger.info('Login')
        logger.info('\tResponse.status' + str(rsp.status))
        logger.info('\tRespond.reason' + str(rsp.reason))
        bts = rsp.read()
        data = json.loads(str(bts, 'utf-8'))
        token = data['URLToken']
    except Exception as e:
        logger.info(f'\nFailed to login\n\t{e}')
        sys.exit(-1)

    # 请求实时快照
    url_snap = f'/api/v1/device/channelsnap?' \
               f'token={token}' \
               f'&serial={opt.serial}' \
               f'&channel={opt.channel}' \
               f'&realtime=true'
    try:
        for i in range(opt.num_test):
            tic = time.time()
            conn.request(method="GET", url=url_snap)
            rsp = conn.getresponse()
            logger.info(f'\nSnap {i}')
            logger.info('\tResponse.status' + str(rsp.status))
            logger.info('\tRespond.reason' + str(rsp.reason))
            bts = rsp.read()
            img = np.asarray(Image.open(io.BytesIO(bts)))
            conn.close()

            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow("image", img)
            cv2.waitKey(1)
    except Exception as e:
        logger.error(e)
        sys.exit(-1)

    logger.info("Done.")
