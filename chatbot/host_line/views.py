from nphard001.web import *
from nphard001.img import *
from nphard001.api_data import *
urlpatterns = []
def _ApplyURLPatterns():
    urlpatterns.append(url(r'webhook', line_webhook))
    urlpatterns.append(url(r'reply', line_reply))
    urlpatterns.append(url(r'identity', identity_view))
    urlpatterns.append(url(r'image', image_fetch_view))
    urlpatterns.append(url(r'test', test_view))
    urlpatterns.append(url(r'^', dump_view))

from linebot import LineBotApi, WebhookParser
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
# NOTE: not safe
line_bot_api = LineBotApi('jn9fH2MHKmg8tpT8oJJj9tsUWefjiUaJyyk2IiHajUQs4g3Du2zcCljM+mi89Acy1ZHZcTBl/Nnyh6EIRSCYnlc96IHI4tEyy0/eKq3XzVyLHMY2ix8FtpVTsJrGbiVPl5kgMcPJPKcmBmeOLi07SwdB04t89/1O/w1cDnyilFU=')
parser = WebhookParser('0b8f65e722157e4be0e36b9228754f56')
info = lambda *args: print('\n[line_webhook]', *args, file=sys.stderr)
# ================================================================
@csrf_exempt
def identity_view(request):
    url_obj = request.GET['url']
    if 'https://linux7.csie.org:3721' not in url_obj:
        return HttpResponseBadRequest()
    info('identity_view got url:', url_obj)
    # mostly it's jpeg, but text/json response also works (hopefully)
    return HttpResponse(HTTPGet(url_obj).content, content_type="image/jpeg")

@csrf_exempt
def line_reply(request):
    json_body = json.loads(request.body.decode('utf-8'))
    info('line_reply, got json:\n'+json.dumps(json_body, indent=1))
    report = {'state': 'done'}
    try:
        if json_body['type']=='text':
            line_bot_api.reply_message(
                json_body['reply_token'],
                TextSendMessage(json_body['text'])
            )
        elif json_body['type']=='image':
            url_small = json_body['url_240x240']
            url_big = json_body.get('url', url_small)
            line_bot_api.reply_message(
                json_body['reply_token'],
                ImageSendMessage(url_small, url_big)
            )
        else:
            return HttpResponseBadRequest()
    except BaseException as e:
        report['state'] = 'error'
        report['BaseException'] = str(e)
    return JsonResponse(report)

@csrf_exempt
def line_webhook(request):
    print('[line_webhook] got request', file=sys.stderr)
    if request.method == 'POST':
        # Line API checking
        signature = request.META['HTTP_X_LINE_SIGNATURE']
        body = request.body.decode('utf-8')
        try:
            events = parser.parse(body, signature)
        except InvalidSignatureError:
            info('InvalidSignatureError')
            return HttpResponseForbidden()
        except LineBotApiError:
            info('LineBotApiError')
            return HttpResponseBadRequest()
        
        # Fork & Response
        if os.fork()>0:
            info('parent process make response')
            return HttpResponse()
        
        for event in events:
            event_repr = repr(event)
            HTTPJson('https://linux7.csie.org:3721/data/line_event', {
                'len': len(events),
                'event_repr': event_repr,
            })
            info('reply_token\n'+event.reply_token)
            # line_bot_api.reply_message(
            #     event.reply_token,
            #     TextSendMessage('event='+event_repr)
            # )
        info('child process done')
        sys.exit(0)
        # return HttpResponse()
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