import cv2
import numpy
from hyperlpr import *
from PIL import Image, ImageDraw, ImageFont


def putCnText(img, text, org, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, numpy.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    draw.text(xy=org, text=text, fill=textColor, font=font)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    input_prefix = "data/"
    output_prefix = "output/"

    os.makedirs(output_prefix, exist_ok=True)
    names = os.listdir(input_prefix)
    names.sort()

    for name in names:
        input_path = os.path.join(input_prefix, name)
        output_path = os.path.join(output_prefix, name)
        img = cv2.imread(input_path)
        if img is None:
            continue

        result = HyperLPR_plate_recognition(img)
        for i in range(len(result)):
            txt, conf, bbox = result[i]
            cv2.rectangle(img, pt1=bbox[:2], pt2=bbox[2:], color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            img = putCnText(img, txt, (bbox[0], bbox[1] - 30), (255, 0, 0), 30)
        cv2.imwrite(output_path, img)
        print(f"{output_path}\n\n")

    print("Done.")
