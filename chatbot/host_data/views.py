from nphard001.web import *
from nphard001.img import *
urlpatterns = []
def _ApplyURLPatterns():
    urlpatterns.append(url(r'image', image_view))
    urlpatterns.append(url(r'test', test_view))
    urlpatterns.append(url(r'^', dump_view))

@csrf_exempt
def image_view(request):
    return RawResponse('WIP')

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