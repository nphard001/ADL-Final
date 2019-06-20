from host_data.views import *
from host_data.models import *
from nphard001.api import *
def _ApplyURLPatterns():
    urlpatterns.append(path(r'line_event', line_event_view))
    urlpatterns.append(path(r'user_text', user_text_view))
    urlpatterns.append(path(r'pending_list', pending_list_view))
    urlpatterns.append(path(r'reply_index', reply_index_view))
# ================================================================
@csrf_exempt
def reply_index_view(request):
    json_body = json.loads(request.body.decode('utf-8'))
    line_userId = json_body['line_userId']
    img_idx = json_body['img_idx']
    token = GetTokenUpdateUserDone(line_userId)
    tosend = {
        'type': 'text',
        'reply_token': token,
        'text': f'train_im index={img_idx}',
    }
    r = HTTPJson('https://nphard001.herokuapp.com/line/reply', tosend)
    return JsonResponse({
        'state': 'done' if r.status_code==200 else 'fail',
        'status_code': r.status_code,
        'reason': r.reason,
    }, safe=False)

@csrf_exempt
def pending_list_view(request):
    json_body = json.loads(request.body.decode('utf-8'))
    pending_list = []
    line_userId = json_body['line_userId'] # INDEV call
    text_list, token = GetUserDialog(line_userId)
    if token:
        pending_list.append({
            'line_userId': line_userId,
            'text_list': text_list,
            'token': token,
        })
    return JsonResponse({
        'state': 'done',
        'pending_list': pending_list,
    }, safe=False)


@csrf_exempt
def line_event_view(request):
    json_body = json.loads(request.body.decode('utf-8'))
    event_dict = json.loads(json_body['event_repr'])
    
    print('--------==== line_event_view ====--------')
    print(json.dumps(event_dict, indent=1))
    if event_dict['source']['type'] != 'user':
        print('not user event, ignored')
        return HttpResponse()
    obj = UserEvent.from_event_dict(event_dict)
    obj.save()
    print('UserEvent object:')
    print(json.dumps(obj.to_dict(), indent=1))
    return HttpResponse()

@csrf_exempt
def user_text_view(request):
    json_body = json.loads(request.body.decode('utf-8'))
    object_set = UserEvent.objects
    object_set = object_set.filter(line_type='message')
    if 'username' in json_body:
        line_userId = UserProfile.objects.filter(username=json_body['username'])
        print(line_userId)
        if line_userId.count()>0:
            line_userId = line_userId[0].line_userId
            object_set = object_set.filter(line_userId=line_userId)
        else:
            object_set = object_set.filter(pk=-1)
    if 'line_userId' in json_body:
        object_set = object_set.filter(line_userId=json_body['line_userId'])
    text_list = []
    token = 'XD'
    for evt_object in object_set.order_by('created'):
        if evt_object.state=='pending':
            token = evt_object.line_replyToken
        txt = evt_object.line_text
        if type(txt) == type('msg'):
            text_list.append(txt)
    return JsonResponse({
        'state': 'done',
        'token': token,
        'text_list': text_list
    }, safe=False)
# ================================================================
_ApplyURLPatterns()