import os
import string
from pathlib import Path

import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import cv2


DEFAULT_FONT = Path(__file__).parent / "fonts/simfang.ttf"


def demo_ocr(image_path: str or Path, image_labels: pd.DataFrame):
    if not os.path.isfile(image_path):
        print(f"Cannot file any file @ {image_path}")
        return None

    image = Image.open(image_path).convert('RGBA')
    overlay = Image.new('RGBA', image.size, (255,255,255,0))
    idraw = ImageDraw.Draw(overlay, 'RGBA')

    for _, (bbox, text) in image_labels[['polygons', 'text']].iterrows():
        bbox = np.array(bbox)
        try:
            text = str(text)
        except:
            text = '###'
        print(text, bbox)
        # font = ImageFont.truetype("arial.ttf", 7)

        if len(bbox) == 4:
            idraw.polygon([tuple(bb) for bb in bbox], fill=(255, 0, 0, 127,), 
                                                   outline=(255, 0, 0, 0),)
        elif len(bbox) == 2:
            idraw.rectangle([tuple(bb) for bb in bbox], fill=(255, 0, 0, 127,), 
                                                     outline=(255, 0, 0, 0),)
        else:
            continue
        idraw.text((bbox.mean(axis=0).astype(int).tolist()), 
                text.encode('latin-1', "ignore"), fill=(255, 0, 0, 255,))

    return Image.alpha_composite(image, overlay).convert('RGB')


def resize_img(img, input_size=600):
    """
    resize image and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img


def draw_box_txt(bbox, text, draw, font, font_size, color):
    # draw ocr results outline
    bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
    draw.rectangle(bbox, fill=color)

    # draw ocr results
    start_y = max(0, bbox[0][1] - font_size)
    tw = font.getsize(text)[0]
    draw.rectangle([(bbox[0][0]+1, start_y), (bbox[0][0]+tw+1, start_y+font_size)], fill=(0, 0, 255))
    draw.text((bbox[0][0] + 1, start_y), text, fill=(255, 255, 255), font=font)


def draw_results_SER(image, ocr_results, font_path=DEFAULT_FONT, font_size=18):
    """
    Draw results of Semantic Entity Recognition [SER]
    """
    np.random.seed(2021)
    color = (np.random.permutation(range(255)),
             np.random.permutation(range(255)),
             np.random.permutation(range(255)),)
    color_map = {
        idx: (color[0][idx], color[1][idx], color[2][idx]) for idx in range(1, 255)
    }
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str) and os.path.isfile(image):
        image = Image.open(image).convert('RGB')
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)

    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    for ocr_info in ocr_results:
        if ocr_info["pred_id"] not in color_map:
            continue
        color = color_map[ocr_info["pred_id"]]
        text = "{}: {}".format(ocr_info["pred"], ocr_info["text"])

        draw_box_txt(ocr_info["bbox"], text, draw, font, font_size, color)

    img_new = Image.blend(image, img_new, 0.5)
    return np.array(img_new)


def draw_results_recognition(image, result, font_path=DEFAULT_FONT, font_size=18):
    np.random.seed(0)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str) and os.path.isfile(image):
        image = Image.open(image).convert('RGB')
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)

    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    color_head = (0, 0, 255)
    color_tail = (255, 0, 0)
    color_line = (0, 255, 0)

    for ocr_info_head, ocr_info_tail in result:
        draw_box_txt(ocr_info_head["bbox"], ocr_info_head["text"], draw, font, font_size, color_head)
        draw_box_txt(ocr_info_tail["bbox"], ocr_info_tail["text"], draw, font, font_size, color_tail)

        center_head = ((ocr_info_head['bbox'][0] + ocr_info_head['bbox'][2]) // 2,
                       (ocr_info_head['bbox'][1] + ocr_info_head['bbox'][3]) // 2,)
        center_tail = ((ocr_info_tail['bbox'][0] + ocr_info_tail['bbox'][2]) // 2,
                       (ocr_info_tail['bbox'][1] + ocr_info_tail['bbox'][3]) // 2,)

        draw.line([center_head, center_tail], fill=color_line, width=5)

    img_new = Image.blend(image, img_new, 0.5)
    return np.array(img_new)


def draw_results_end2end(dt_boxes, strs, img_path):
    src_im = cv2.imread(img_path)
    for box, str in zip(dt_boxes, strs):
        box = box.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        cv2.putText(src_im, str, org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=1)
    return src_im


def draw_results_detection(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


def draw_results_ocr(image, boxes, txts=None, scores=None, drop_score=0.5, font_path=DEFAULT_FONT):
    """
    Visualize the results of OCR detection and recognition

    Parameters
    ----------
    image(Image|array): RGB image
    boxes(list): boxes with shape(N, 4, 2)
    txts(list): the texts
    scores(list): txxs corresponding scores
    drop_score(float): only scores greater than drop_threshold will be visualized
    font_path: the path of font which is used to draw text
    
    Returns
    -------
    the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(txts, scores, img_h=img.shape[0], img_w=600, threshold=drop_score, font_path=font_path)
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image


def draw_ocr_box_txt(image, boxes, txts, scores=None, drop_score=0.5, font_path=DEFAULT_FONT):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon([box[0][0], box[0][1], 
                            box[1][0], box[1][1], 
                            box[2][0], box[2][1], 
                            box[3][0], box[3][1],], outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)

        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text((box[0][0]+3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def text_visual(texts, scores, img_h=400, img_w=600, threshold=0., font_path=DEFAULT_FONT):
    """
    create new blank img and draw txt on it

    Parameters
    ----------
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    
    Returns
    -------
    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)

        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1

    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)


def str_count(s):
    """
    Count 
        the number of Chinese characters, 
        a single English character, and 
        a single number equal to half the length of Chinese characters.

    Parameters
    ----------
        s(string): the input of string
    
    Returns
    -------
        the number of Chinese characters
    """
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


