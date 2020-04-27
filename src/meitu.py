# coding: utf-8
import base64

import cv2
import numpy as np
import requests


AUTHORIZATION = '6qqkLDqc18/fSetHOqw/YeYsc9cuaz13Tkt6dFVWdVhFaUhTTmZENEEwNlNHd3FYYXR6VXZTMCZ0PTE1ODc5NDkwNDYmcj03Mjg3NTY0ODg0Njk1NTY0NjM4JmY9'


def imencode(im):
    if isinstance(im, np.ndarray):
        ret, im = cv2.imencode('.jpg', im)
        assert ret
    elif isinstance(im, str):
        with open(im, 'rb') as fp:
            im = fp.read()
    else:
        im.seek(0)
        im = im.read()
    return base64.b64encode(im).decode()


def imdecode(im):
    im = base64.b64decode(im)
    im = np.asarray(bytearray(im), dtype='uint8')
    return cv2.imdecode(im, -1)


def imwrite(fn, im):
    im = base64.b64decode(im)
    with open(fn, 'wb') as fp:
        fp.write(im)


def post(func, data):
    url = f'https://openapi.mtlab.meitu.com/v1/{func}'
    headers = {
        'Authorization': AUTHORIZATION,
        'AuthorizationType': '1',
    }
    try:
        result = requests.post(url, json=data, headers=headers).json()
    except requests.exceptions.ConnectionError:
        result = {}
    # 有可能返回空
    return result if 'media_info_list' in result else None


def hair_segment(file):
    # http://ai.meitu.com/doc/?id=16
    func = 'segmentEx'
    data = {
        'parameter': {
            'outputType': 1     # 输出头发的mask图
        },
        'media_info_list': [{
            'media_data': imencode(file),
            'media_profiles': {'media_data_type': 'jpg'},
        }],
    }
    result = post(func, data)
    im = result['media_info_list'][0]['media_data']
    return im


def photo_segment(file):
    # http://ai.meitu.com/doc/?id=9
    func = 'PhotoSegment'
    data = {
        'media_info_list': [{       # 可多个
            'media_data': imencode(file),
            'media_profiles': {'media_data_type': 'jpg'}
        }],
    }
    result = post(func, data)
    im = result['media_info_list'][0]['media_data']
    return im


if __name__ == '__main__':
    im = cv2.imread('t.jpg')
    mask = photo_segment(im)
    mask = imdecode(mask)
    cv2.imwrite('t.out.6.png', mask)
