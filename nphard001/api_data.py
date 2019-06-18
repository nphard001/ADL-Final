import requests
import warnings
from typing import Optional, Dict
warnings.filterwarnings('ignore', 'Unverified HTTPS request is being made.')

from PIL import Image
from io import BytesIO
def Request2Image(request_obj):
    r'''convert request byte content into PIL image object'''
    return Image.open(BytesIO(request_obj.content))

class HostData:
    def __init__(self, host: str=r'https://linux7.csie.org:3721'):
        host = host.rstrip('/')
        self.host = host
    def get_metadata(self):
        return HTTPJson2Json(f'{self.host}/data/attributedata', {
            'type': 'metadata',
        })
    def get_txt(self, category: str='earrings_studs', img_id: int=1555):
        return HTTPJson2Json(f'{self.host}/data/attributedata', {
            'type': 'txt',
            'category': category,
            'img_id': img_id,
        })
    def get_jpg(self, category: str='earrings_studs', img_id: int=1555):
        return HTTPJson2Image(f'{self.host}/data/attributedata', {
            'type': 'jpg',
            'category': category,
            'img_id': img_id,
        })
        

def HTTPPost(url: str, params: Optional[Dict]=None):
    r'''call requests.post without CA verify'''
    if params:
        return requests.post(
            url, params, verify=False)
    else:
        return requests.post(
            url, verify=False)
def HTTPGet(url: str, params: Optional[Dict]=None):
    r'''call requests.get without CA verify'''
    if params:
        return requests.get(
            url, params, verify=False)
    else:
        return requests.get(
            url, verify=False)
def HTTPJson(url: str, jsonable_object):
    r'''send post with json'''
    return requests.post(url, json=jsonable_object, verify=False)

def HTTPPost2Text(url: str, params: Optional[Dict]=None):
    return HTTPPost(url, params).text
def HTTPPost2Json(url: str, params: Optional[Dict]=None):
    return HTTPPost(url, params).json()
def HTTPPost2Image(url: str, params: Optional[Dict]=None):
    return Request2Image(HTTPPost(url, params))
def HTTPJson2Text(url: str, jsonable_object):
    return HTTPJson(url, jsonable_object).text
def HTTPJson2Json(url: str, jsonable_object):
    return HTTPJson(url, jsonable_object).json()
def HTTPJson2Image(url: str, jsonable_object):
    return Request2Image(HTTPJson(url, jsonable_object))
def HTTPGet2Text(url: str, params: Optional[Dict]=None):
    return HTTPGet(url, params).text
def HTTPGet2Json(url: str, params: Optional[Dict]=None):
    return HTTPGet(url, params).json()
def HTTPGet2Image(url: str, params: Optional[Dict]=None):
    return Request2Image(HTTPGet(url, params))