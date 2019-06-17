from nphard001.web import *
from nphard001.img import *
from nphard001.api_data import *
urlpatterns = []
def _ApplyURLPatterns():
    urlpatterns.append(url(r'webhook', line_webhook))
    urlpatterns.append(url(r'image', image_fetch_view))
    urlpatterns.append(url(r'test', test_view))
    urlpatterns.append(url(r'^', dump_view))

from linebot import LineBotApi, WebhookParser
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
# NOTE: not safe
line_bot_api = LineBotApi('jn9fH2MHKmg8tpT8oJJj9tsUWefjiUaJyyk2IiHajUQs4g3Du2zcCljM+mi89Acy1ZHZcTBl/Nnyh6EIRSCYnlc96IHI4tEyy0/eKq3XzVyLHMY2ix8FtpVTsJrGbiVPl5kgMcPJPKcmBmeOLi07SwdB04t89/1O/w1cDnyilFU=')
parser = WebhookParser('0b8f65e722157e4be0e36b9228754f56')
@csrf_exempt
def line_webhook(request):
    print('[line_webhook] got request', file=sys.stderr)
    if request.method == 'POST':
        signature = request.META['HTTP_X_LINE_SIGNATURE']
        body = request.body.decode('utf-8')
        try:
            events = parser.parse(body, signature)
        except InvalidSignatureError:
            print('[line_webhook] InvalidSignatureError', file=sys.stderr)
            return HttpResponseForbidden()
        except LineBotApiError:
            print('[line_webhook] LineBotApiError', file=sys.stderr)
            return HttpResponseBadRequest()
        for event in events:
            # multi-event single reply
            line_bot_api.reply_message(
                event.reply_token,
                ImageSendMessage(
                    original_content_url='https://nphard001.herokuapp.com/line/image',
                    preview_image_url='https://nphard001.herokuapp.com/line/image',
                )
            )
            break
            # print("""#---== got new EVENT!!! ==---#\n%s\n#---== --- ==---#"""%repr(event))
            # if isinstance(event, MessageEvent):
            #     line_bot_api.reply_message(
            #         event.reply_token,
            #         TextSendMessage(text=json.dumps(event, indent=1))
            #     )
        return HttpResponse()
    else:
        return HttpResponseBadRequest()

@never_cache
@csrf_exempt
def image_fetch_view(request):
    # url = 'https://linux7.csie.org:3721/static/attributedata/bags_evening/0/img_bags_evening_216.jpg'
    url = 'https://linux7.csie.org:3721/data/image'
    return HttpResponse(HTTPGet(url).content, content_type="image/jpeg")

@csrf_exempt
def test_view(request):
    # JSON
    json_body = json.loads(request.body.decode('utf-8'))
    return JsonResponse({
        'state': 'done', 
        'id_list': [1, 3, 3, 7],
        'json_body': json_body,
        'method': request.method,
        })
    
@csrf_exempt
def dump_view(request):
    dump_content = json.dumps({
        'GET': dict(request.GET),
        'POST': dict(request.POST),
    }, indent=1)
    print('[host_line] dump_content')
    print(dump_content)
    return RawResponse(dump_content)

# ================================================================
_ApplyURLPatterns()