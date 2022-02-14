import numpy
import sys
from model.hyperlpr import *
from PIL import Image, ImageDraw, ImageFont


def putCnText(img, text, org, font, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, numpy.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)
    img_font = ImageFont.truetype(font, textSize, encoding="utf-8")
    draw.text(xy=org, text=text, fill=textColor, font=img_font)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    input_prefix = "data/"
    output_prefix = "output/"
    checkpoint_folder = "checkpoint/"
    # 中文字体，linux系统下可能需要自行安装ttf字体
    font = "/usr/share/fonts/win10/STFANGSO.TTF" if sys.platform == "linux" else "font/simsun.ttc"

    os.makedirs(output_prefix, exist_ok=True)
    names = os.listdir(input_prefix)
    names.sort()

    # 车牌识别模型，caffe model
    PR = LPR(checkpoint_folder)

    for name in names:
        input_path = os.path.join(input_prefix, name)
        output_path = os.path.join(output_prefix, name)
        img = cv2.imread(input_path)
        if img is None:
            continue

        result = PR.plate_recognition(img, minSize=30, charSelectionDeskew=True)
        for i in range(len(result)):
            txt, conf, bbox = result[i]     # 字符，自信度，位置
            cv2.rectangle(img, pt1=bbox[:2], pt2=bbox[2:], color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            img = putCnText(img, txt, (bbox[0], bbox[1] - 30), font, (255, 0, 0), 30)
        cv2.imwrite(output_path, img)
        print(f"{output_path}\n\n")

    print("Done.")
