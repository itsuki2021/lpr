from http.client import HTTPConnection
from model.hyperlpr import *
from PIL import Image, ImageDraw, ImageFont
import json
import hashlib
import time
import cv2
import io
import sys


def encrypt(s: str) -> str:
    enc = hashlib.md5()
    enc.update(s.encode('utf-8'))
    return (enc.hexdigest()).lower()


def putCnText(img, text, org, font, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)
    img_font = ImageFont.truetype(font, textSize, encoding="utf-8")
    draw.text(xy=org, text=text, fill=textColor, font=img_font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    # 参数
    host = '8.141.69.166'                                       # 主机地址
    port = 10256                                                # 端口
    usr = 'admin'                                               # 用户名
    pwd = 'admin'                                               # 密码
    method = 'GET'                                              # 请求方法
    serial = '34020000001110000031'                             # 摄像头设备号
    channel = 1                                                 # 设备通道
    checkpoint_folder = "checkpoint/"                           # 模型路径
    font = "/usr/share/fonts/win10/STFANGSO.TTF" \
        if sys.platform == "linux" else "font/simsun.ttc"       # 中文字体，linux系统下需要自行安装STFANGSO.TTF字体
    num_test = 10                                               # 测试次数

    # 登录，获取token
    url_login = f'/api/v1/login?' \
                f'username={usr}&password={encrypt(pwd)}'
    headers = {'Content-Type': 'text/html;charset=utf-8'}
    conn = HTTPConnection(host=host, port=port)
    conn.request(method=method, url=url_login, headers=headers)
    rsp = conn.getresponse()
    print('Login')
    print('\tResponse.status' + str(rsp.status))
    print('\tRespond.reason' + str(rsp.reason))
    bts = rsp.read()
    data = json.loads(str(bts, 'utf-8'))
    token = data['URLToken']

    # 请求实时快照
    url_snap = f'/api/v1/device/channelsnap?' \
               f'token={token}' \
               f'&serial={serial}' \
               f'&channel={channel}' \
               f'&realtime=true'
    PR = LPR(checkpoint_folder)
    for i in range(num_test):
        tic = time.time()
        conn.request(method=method, url=url_snap)
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
            txt, conf, bbox = result[i]     # 字符，自信度，位置
            cv2.rectangle(img, pt1=bbox[:2], pt2=bbox[2:], color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            img = putCnText(img, txt, (bbox[0], bbox[1] - 30), font, (255, 0, 0), 30)
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", img)
        cv2.waitKey(2)

    print("Done.")
