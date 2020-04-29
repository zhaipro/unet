import json

import requests
# https://github.com/aliyun/aliyun-openapi-python-sdk
from aliyunsdkcore.client import AcsClient
from aliyunsdkimageseg.request.v20191230.ParseFaceRequest import ParseFaceRequest


ID = 'unknown'
SECRET = 'unknown'
CLIENT = AcsClient(ID, SECRET, 'cn-shanghai')


def parse_face(fn, path='zhaipro'):
    url = f'http://{path}.oss-cn-shanghai.aliyuncs.com{fn}'
    request = ParseFaceRequest()
    request.set_accept_format('json')
    request.set_ImageURL(url)
    r = CLIENT.do_action_with_exception(request)
    return json.loads(r)


if __name__ == '__main__':
    fn = '/unet/screenshots/14.jpg'
    r = parse_face(fn)
    print(r)
