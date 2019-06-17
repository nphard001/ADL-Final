from nphard001.web import *
from nphard001.img import *
from nphard001.api_data import *
from django.contrib.staticfiles.templatetags.staticfiles import static
urlpatterns = []
def _ApplyURLPatterns():
    urlpatterns.append(url(r'image', image_view))
    urlpatterns.append(url(r'test', test_view))
    urlpatterns.append(url(r'^', dump_view))

@csrf_exempt
def image_view(request):
    # return 240x240 image
    img = HTTPGet2Image('https://linux7.csie.org:3721/static/attributedata/bags_evening/0/img_bags_evening_216.jpg')
    img.thumbnail([240, 240]) # it's inplace
    img_bytes = BytesIO()
    img.save(img_bytes, 'jpeg')
    return HttpResponse(img_bytes.getvalue(), content_type="image/jpeg")
    
    # forward image from static
    # img_req = HTTPGet(static('attributedata/bags_evening/0/img_bags_evening_216.jpg'))
    # return HttpResponse(img_req.content) # get bytes & send bytes

@csrf_exempt
def test_view(request):
    d1 = dict(request.GET)
    d1['array'] = [1, 3, 3, 7]
    d1['attached_text'] = TestClass().text()
    d1['attached_image'] = TestClass().image()
    d2 = dict(request.POST)
    d2['post_len'] = len(d2.keys())
    s0 = json.dumps({
        'GET': d1,
        'POST': d2,
    })
    return RawResponse(s0)

@csrf_exempt
def dump_view(request):
    dump_content = json.dumps({
        'GET': dict(request.GET),
        'POST': dict(request.POST),
    }, indent=1)
    print('[host_data] dump_content')
    print(dump_content)
    return RawResponse(dump_content)

# ================================================================
_ApplyURLPatterns()