from nphard001.utils import *
from nphard001.web import *
from PIL import Image, ImageDraw
from io import BytesIO
def ConvertThumbnail(img, request=None):
    if type(request)==type(None):
        img.thumbnail([240, 240])
        return img
    tsize = int(RequestGET(request, 'size') or 240)
    text = str(RequestGET(request, 'text'))
    img.thumbnail([tsize, tsize])
    if text!='None':
        img = img.convert('RGBA')
        img_txt = Image.new('RGBA', [32, 32], (255,255,255,50))
        ImageDraw.Draw(img_txt).text((3, 0), text, (0, 0, 0))
        img_txt = img_txt.resize(img.size)
        img = Image.alpha_composite(img, img_txt)
        img = img.convert('RGB')
    return img
