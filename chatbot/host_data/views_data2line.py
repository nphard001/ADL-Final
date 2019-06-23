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
    r'''reply image index from GPU server'''
    json_body = json.loads(request.body.decode('utf-8'))
    line_userId = json_body['line_userId']
    img_idx = json_body['img_idx']
    token = GetTokenUpdateUserDone(line_userId)
    print('token:', token)
    if type(token)!=type('3ae07b'):
        print('fail')
        return JsonResponse({'state': 'fail', 'msg': 'no token (already replied that user?)'}, safe=False)
    url_240x240, url = GetTrainURLByIndex(img_idx)
    tosend = {
        'type': 'image',
        'reply_token': token,
        'url_240x240': url_240x240,
        'url': url
    }
    print('reply_index_view, tosend:')
    print(json.dumps(tosend, indent=1))
    r = HTTPJson('https://nphard001.herokuapp.com/line/reply', tosend)
    return_json = {
        'state': 'done' if r.status_code==200 else 'fail',
        'status_code': r.status_code,
        'reason': r.reason,
    }
    UserReplyImage(
        line_userId=line_userId,
        train_idx=img_idx,
        info=json.dumps({
            'tosend': tosend,
            'return_json': return_json
        }),
    ).save()
    return JsonResponse(return_json, safe=False)


@never_cache
@csrf_exempt
def pending_list_view(request):
    r'''return data to GPU server'''
    # DO NOT ignore the request!!!
    _ = json.loads(request.body.decode('utf-8'))
    pending_list = GetPendingList()
    return JsonResponse({
        'state': 'done',
        'pending_list': pending_list,
    }, safe=False)


@csrf_exempt
def line_event_view(request):
    r'''get line event from heroku'''
    json_body = json.loads(request.body.decode('utf-8'))
    event_dict = json.loads(json_body['event_repr'])
    
    print('--------==== line_event_view ====--------')
    print(json.dumps(event_dict, indent=1))
    if event_dict['source']['type'] != 'user':
        print('not user event, ignored')
        return HttpResponse()
    obj = UserEvent.create_from_event_dict(event_dict)
    print('UserEvent object:')
    print(json.dumps(obj.to_dict(), indent=1))
    return HttpResponse()

@csrf_exempt
def user_text_view(request):
    r'''(deprecated) find text_list related to some user'''
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