import base64
import numpy as np
from PIL import Image
import os
import os.path as osp
import cv2
import shutil

try:
    import accimage
except ImportError:
    accimage = None


def cv2_base64encode(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    _, buffer = cv2.imencode('.jpg', image, encode_param)
    return base64.b64encode(buffer.tostring())


def cv2_base64decode(base64_string):
    nparr = np.fromstring(base64.b64decode(base64_string), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def cv2_PIL(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def crop(img, xmin, ymin, xmax, ymax):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((xmin, ymin, xmax, ymax))


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def extract_vid(src_path, out_dir, period=1):
    assert osp.exists(src_path), 'Source file not found !'
    vid = cv2.VideoCapture(src_path)
    if osp.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    cnt = 0
    while vid.isOpened():
        _, img = vid.read()
        if img is None: break
        if cnt % period == 0:
            img_name = '{:05n}.jpg'.format(cnt)
            img_path = osp.join(out_dir, img_name)
            cv2.imwrite(img_path, img)
            cnt += 1
    vid.release()


def get_public_ip():
    from requests import get
    # Get external ip for host machine
    return get('https://api.ipify.org').text

def draw_polygon(img, box, colour=(0, 0, 255), thickness=2):
    '''
        :param img: cv2 image
        :param box: list vertices of box [(x1, y1), (x2, y2, ... , (xn, yn)]
    '''
    cv2.polylines(img, [np.asarray(box, np.int32)], True, colour, thickness=thickness)


def plot_bbox(img, bboxes, colours):
    for d in bboxes:
        if d[-1] > 0:
            color = colours[int(d[-1]) % 16].tolist()
            tl = round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
            c1, c2 = (int(d[0]), int(d[1])), (int(d[2]), int(d[3]))
            cv2.rectangle(img, c1, c2, color, thickness=tl)
            # Plot score
            tf = max(tl - 1, 1)  # font thickness
            if d[-1] < 1: label = 'basket'
            else: label = '%d' % int(d[-1])     # local_id
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [0, 0, 0], thickness=tf,
                        lineType=cv2.LINE_AA)

def plot_tracjectories(img, pts, colours):
    # Plot trajectories
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        for j in range(0, len(pts[i - 1])):
            for k in range(0, len(pts[i])):
                if (pts[i - 1][j][2] == pts[i][k][2]) and (pts[i - 1][j][2] > 0):
                    color = colours[pts[i - 1][j][2] % 16].tolist()
                    cv2.line(img, pts[i - 1][j][0:2], pts[i][k][0:2], color, thickness=2)
                    continue