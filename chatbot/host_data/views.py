import linecache
from nphard001.web import *
from nphard001.img import *
from nphard001.api_data import *
from nphard001.attr_parser import AttrParser
from host_data.models import AttrMetadata
import numpy as np
from django.contrib.staticfiles.templatetags.staticfiles import static
_MakoMgr = MakoManager(['mako'])
_Mako = _MakoMgr.render
urlpatterns = []
def _ApplyURLPatterns():
    urlpatterns.append(path(r'random/<int:N>/<int:M>', random_view))
    urlpatterns.append(path(r'image/<slug:img_type>/train/<int:train_idx>', image_train_view))
    urlpatterns.append(path(r'image/<slug:img_type>/<slug:ctg>/<int:img_id>', image_view))
    urlpatterns.append(url(r'attributedata', attributedata_view))

@csrf_exempt
def random_view(request, N=18, M=3):
    object_list = []
    api = HostDataAPI()
    meta = api.get_metadata()
    ctg = 'womens_high_heels'
    for x in np.random.choice(list(range(1, 1670+1)), size=N):
        img_id = int(x)
        img = api.get_jpg(ctg, img_id)
        object_list.append([
            f'https://linux7.csie.org:3721/data/image/raw/{ctg}/{img_id}', 
            api.get_txt(ctg, img_id)])
    
    # ---
    title = 'random_view'
    nav = get_common_nav()
    body = HTMLImageTable(object_list, num_each_row=M)
    
    # collect (raw, context)
    raw = '\n'.join([
        r'<%inherit file="basic.html"/>',
    ]).strip()
    context = locals().copy()
    response = HttpResponse(ChatbotMako(raw, context).encode('utf-8'))
    return response

@csrf_exempt
def attributedata_view(request):
    # JSON
    json_body = json.loads(request.body.decode('utf-8'))
    if json_body['type']=='metadata':
        ctg_list = AttrMetadata.objects.values_list('category', flat=True).distinct()
        ctg_list = list(ctg_list)
        lod = []
        for ctg in ctg_list:
            ctg_objs = AttrMetadata.objects.filter(category=ctg).order_by('img_id')
            lod.append({
                'category': ctg,
                'num': ctg_objs.count(),
                'min_id': ctg_objs.first().img_id,
                'max_id': ctg_objs.last().img_id,
            })
        return JsonResponse({
            'state': 'done',
            'target': lod,
            'ctg_list': ctg_list
        })
    elif json_body['type']=='jpg':
        category = json_body['category']
        img_id = json_body['img_id']
        obj = AttrMetadata.objects.filter(category=category, img_id=img_id)[0]
        with open(obj.img_path, 'rb') as f:
            return HttpResponse(f.read(), content_type="image/jpeg")
    elif json_body['type']=='txt':
        category = json_body['category']
        img_id = json_body['img_id']
        obj = AttrMetadata.objects.filter(category=category, img_id=img_id)[0]
        with open(obj.dsr_path, 'r') as f:
            txt = f.read().strip()
        return JsonResponse({
            'state': 'done',
            'target': txt,
        })
    else:
        return JsonResponse({
            'state': 'fail',
            'msg': 'no such type'
        })

@cache_page(60 * 60 * 24)
@csrf_exempt
def image_train_view(request, img_type: str, train_idx: int):
    if any([img_type not in ['raw', 'thumbnail'],
        train_idx<0 or train_idx>9999]):
        return HttpResponseBadRequest()
    filename = linecache.getline('fashion_retrieval/dataset/train_im_names.txt', train_idx+1)
    ctg, img_id = AttrFilenameSplit(filename)
    img_id = int(img_id)
    return HttpResponse(AttrMetadata.GetImgBinary(img_type, ctg, img_id), content_type="image/jpeg")

@cache_page(60 * 60 * 24)
@csrf_exempt
def image_view(request, img_type:str='raw', ctg: str='earrings_drop', img_id: int=1694):
    if img_type not in ['raw', 'thumbnail']:
        return HttpResponseBadRequest()
    return HttpResponse(AttrMetadata.GetImgBinary(img_type, ctg, img_id), content_type="image/jpeg")

# ================================================================
_ApplyURLPatterns()