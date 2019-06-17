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

def HTTPPost2Text(url: str, params: Optional[Dict]=None):
    return HTTPPost(url, params).text
def HTTPPost2Json(url: str, params: Optional[Dict]=None):
    return HTTPPost(url, params).json()
def HTTPPost2Image(url: str, params: Optional[Dict]=None):
    return Request2Image(HTTPPost(url, params))
def HTTPGet2Text(url: str, params: Optional[Dict]=None):
    return HTTPGet(url, params).text
def HTTPGet2Json(url: str, params: Optional[Dict]=None):
    return HTTPGet(url, params).json()
def HTTPGet2Image(url: str, params: Optional[Dict]=None):
    return Request2Image(HTTPGet(url, params))