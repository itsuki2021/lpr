from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


def putCnText(img, text, org, font, textColor=(0, 255, 0), textSize=20):
    """ 图像绘制中文

    :param img:         输入图像
    :param text:        绘制文本
    :param org:         绘制位置
    :param font:        字体
    :param textColor:   文本颜色
    :param textSize:    文本大小
    :return:            绘制后的图像
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)
    img_font = ImageFont.truetype(font, textSize, encoding="utf-8")
    draw.text(xy=org, text=text, fill=textColor, font=img_font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)