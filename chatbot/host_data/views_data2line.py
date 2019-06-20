from host_data.views import *
from host_data.models import UserEvent
def _ApplyURLPatterns():
    urlpatterns.append(path(r'line_event', line_event_view))
# ================================================================
@csrf_exempt
def line_event_view(request):
    json_body = json.loads(request.body.decode('utf-8'))
    event = json_body['event_repr']
    print('--------==== line_event_view ====--------')
    obj = UserEvent(
        # line_userId = json_body['event']['source']['userId'],
        # line_replyToken = json_body['event']['replyToken'],
        line_event = json_body['event_repr']
        )
    obj.save()
    print(json.dumps(json_body, indent=1))
    return HttpResponse()
r'''
{
 "message": {
  "id": "10069058180662",
  "packageId": "7920494",
  "stickerId": "195012613",
  "type": "sticker"
 },
 "replyToken": "b2ea723f79244cc59478d6bcd96f6bbf",
 "source": {
  "type": "user",
  "userId": "U9d6503da59b6b0d5efba3c7af7af5125"
 },
 "timestamp": 1560944438172,
 "type": "message"
}

'''
# ================================================================
_ApplyURLPatterns()