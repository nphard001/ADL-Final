from chatbot_main.utils import *
from chatbot_main.view_basic import *
from chatbot_main.model import *
def _Response(x: str):
    return HttpResponse(x.encode('utf-8'))
def _Int(dic, k, dft):
    if k in dic:
        return int(dic[k][0])
    return dft
def _Float(dic, k, dft):
    if k in dic:
        return float(dic[k][0])
    return dft
def _Str(dic, k, dft):
    if k in dic:
        return str(dic[k][0])
    return dft
################
from linebot import LineBotApi, WebhookParser
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
# NOTE: not safe
line_bot_api = LineBotApi('jn9fH2MHKmg8tpT8oJJj9tsUWefjiUaJyyk2IiHajUQs4g3Du2zcCljM+mi89Acy1ZHZcTBl/Nnyh6EIRSCYnlc96IHI4tEyy0/eKq3XzVyLHMY2ix8FtpVTsJrGbiVPl5kgMcPJPKcmBmeOLi07SwdB04t89/1O/w1cDnyilFU=')
parser = WebhookParser('0b8f65e722157e4be0e36b9228754f56')
from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def chatbot_callback(request):
    print('[Chatbot] new chatbot_callback request')
    print(json.dumps({
        'GET': dict(request.GET),
        'POST': dict(request.POST),
    }, indent=1))
    if request.method == 'POST':
        signature = request.META['HTTP_X_LINE_SIGNATURE']
        body = request.body.decode('utf-8')
        try:
            events = parser.parse(body, signature)
        except InvalidSignatureError:
            return HttpResponseForbidden()
        except LineBotApiError:
            return HttpResponseBadRequest()
        for event in events:
            print("""#---== got new EVENT!!! ==---#\n%s\n#---== --- ==---#"""%repr(event))
            if isinstance(event, MessageEvent):
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=json.dumps(event, indent=1))
                )
        return HttpResponse()
    else:
        return HttpResponseBadRequest()
    return dump_view(request)
################
def dump_view(request):
    dump_content = json.dumps({
        'GET': dict(request.GET),
        'POST': dict(request.POST),
    }, indent=1)
    print('[Chatbot] dump_content')
    print(dump_content)
    return _Response(dump_content)
    
# def API_pos(request):
#     r'''add predicted position'''
#     dict_get = dict(request.GET)
#     if 'mac' not in dict_get:
#         return _Response('[ERROR] which mac address?')
#     if 'x' not in dict_get or 'y' not in dict_get:
#         return _Response('[ERROR] what x, y?')
#     mac_get = _Str(dict_get, 'mac', 'nopi')
#     tag_get = _Str(dict_get, 'tag', '')
#     x_get = _Float(dict_get, 'x', 0)
#     y_get = _Float(dict_get, 'y', 0)
#     print('--------==== API_pos ====--------')
#     find_dup = PosPredict.objects.filter(mac=mac_get, tag=tag_get)
#     if len(find_dup)>=1:
#         toadd = find_dup[0]
#         print('dup!', toadd)
#         toadd.x = x_get
#         toadd.y = y_get
#         toadd.save()
#         return _Response('modified')
#     toadd = PosPredict(
#         mac=mac_get, 
#         tag=tag_get,
#         x=x_get,
#         y=y_get,)
#     toadd.save()
#     return _Response('done')

# def API_raw(request):
#     dict_get = dict(request.GET)
#     if 'pi' not in dict_get:
#         return _Response('[ERROR] which pi?')
#     if 'c' not in dict_get:
#         return _Response('[ERROR] what c? (csv content?)')
#     toadd = RawData(
#         pi=_Str(dict_get, 'pi', 'nopi'), 
#         target=_Str(dict_get, 'c', 'empty'))
#     toadd.save()
#     return _Response('done')
# def API_parsed(request):
#     return_str = ''
#     dict_get = dict(request.GET)
#     print(dict_get)
#     maxn = _Int(dict_get, 'maxn', 10)
#     lod = []
#     for raw_data in RawData.objects.order_by('-created')[:maxn]:
#         lod.append(raw_data.to_dict())
#     return _Response(json.dumps(lod, indent=1))

# def ExampleParser(s0):
#     return s0
