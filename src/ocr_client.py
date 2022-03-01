from src.utils.img_slice import get_slice_windows, is_overlapped, bbox_union
from src.utils.img_plot import putCnText
from loguru import logger
import cv2
import numpy as np
import requests


def draw_result(img: np.ndarray, recog_result):
    img_copy = img.copy()
    for result in recog_result:
        box = result['textdet']
        text, score = result['textrecog']
        cv2.rectangle(img=img_copy,
                      pt1=[int(v) for v in box[:2]],
                      pt2=[int(v) for v in box[2:4]],
                      color=(0, 255, 0),
                      thickness=4)
        try:
            img_copy = putCnText(img=img_copy,
                                 text=f'{text}|{score:.2f}',
                                 org=(int(box[0]), int(box[1]) - 40),
                                 font='font/simsun.ttc',
                                 textColor=(0, 255, 0),
                                 textSize=40)
        except Exception as e:
            logger.error(e)

    return img_copy


class OCRClient:
    def __init__(self, det_model='dbnet', recog_model='sar_cn',
                 inference_addr='0.0.0.0:8080'):
        self.url_det = 'http://' + inference_addr + '/predictions/' + det_model
        self.url_recog = 'http://' + inference_addr + '/predictions/' + recog_model

    def __slice_text_det(self, img: np.ndarray,
                         slice_height=800,
                         slice_width=800,
                         slice_height_overlap=0.1,
                         slice_width_overlap=0.1,
                         threshold_det=0.5,
                         **kwargs):
        logger.info("slice text detecting...")
        windows = get_slice_windows(input_h=img.shape[0], input_w=img.shape[1],
                                    sli_h=slice_height, sli_w=slice_width,
                                    olp_h_ratio=slice_height_overlap,
                                    olp_w_ratio=slice_width_overlap)
        bboxes = np.ndarray(shape=(0, 5), dtype=float)  # object bboxes
        for j, win in enumerate(windows):
            logger.info(f"\t\twindows: {j + 1}/{len(windows)}")
            data = cv2.imencode('.jpg', img[win[1]:win[3], win[0]:win[2], :])[1].tobytes()
            response = requests.post(self.url_det, data)
            boundary_result = response.json()['boundary_result']
            for poly in boundary_result:
                score_det = poly[-1]
                if score_det < threshold_det:
                    logger.info(f"Not enough detect confidence. recog {score_det} < threshold {threshold_det}")
                    continue
                poly = [int(v) for v in poly]
                poly = [int(poly[k]) + win[0] if k % 2 == 0 else int(poly[k]) + win[1] for k in range(len(poly) - 1)]
                xx = poly[0::2]
                yy = poly[1::2]
                box = np.array([min(xx), min(yy), max(xx), max(yy), score_det], dtype=float).reshape(1, 5)
                bboxes = np.concatenate((bboxes, box), axis=0)
        logger.info(f"number of objects: {len(bboxes)}")

        # merge bboxes
        merged_bboxes = np.ndarray(shape=(0, 5), dtype=float)
        for box in bboxes:
            olp_indices = []    # indices of overlapped bboxes
            for idx in range(len(merged_bboxes)):
                if is_overlapped(merged_bboxes[idx], box):
                    olp_indices.append(idx)
            new_box = bbox_union(np.concatenate((merged_bboxes[olp_indices], box.reshape(1, 5)), axis=0))
            merged_bboxes = np.delete(merged_bboxes, olp_indices, axis=0)
            merged_bboxes = np.concatenate((merged_bboxes, new_box), axis=0)

        logger.info(f"number of objects after merging: {len(merged_bboxes)}")

        return merged_bboxes

    def __text_recog(self, img: np.ndarray,
                     bboxes: np.ndarray,
                     threshold_recog=0.5,
                     **kwargs):
        # slice recognition
        logger.info("text recognizing...")
        recog_result = []
        for j, box in enumerate(bboxes):
            logger.info(f"\t\tbox: {j + 1}/{len(bboxes)}")
            data_lp = cv2.imencode('.jpg', img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :])[1].tobytes()
            response = requests.post(self.url_recog, data_lp)
            text, score_ocr = response.json()['text'], response.json()['score']
            if score_ocr < threshold_recog:
                logger.info(f"Not enough detect confidence. recog {score_ocr} < threshold {threshold_recog}")
                continue

            recog_result.append({
                "textdet": list(box),
                "textrecog": [text, score_ocr]
            })
        logger.info(f"number of word: {len(recog_result)}")

        return recog_result

    def read_text(self, img: np.ndarray, draw=False, **kwargs):
        bboxes = self.__slice_text_det(img, **kwargs)
        recog_result = self.__text_recog(img, bboxes, **kwargs)
        if draw:
            img_draw = draw_result(img, recog_result)
            return recog_result, img_draw
        else:
            return recog_result

