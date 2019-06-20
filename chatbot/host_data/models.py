from nphard001.django_model import *

class AttrMetadata(BaseModel):
    category = models.TextField(db_index=True, default='unknown')
    img_id = models.IntegerField(db_index=True, default=-1)
    img_path = models.TextField(default='None')
    dsr_path = models.TextField(default='None')

class UserProfile(BaseModel):
    updated = models.DateTimeField(auto_now=True, db_index=True)
    dialog_end = models.DateTimeField(default=timezone.now, db_index=True)
    username = models.TextField(default='None', db_index=True)
    line_userId = models.TextField(default='None', db_index=True)
    mode = models.TextField(default='auto', db_index=True) # auto / debug
    pk_last_pending = models.IntegerField(default=0, db_index=True)

class UserEvent(BaseModel):
    created = models.DateTimeField(auto_now_add=True, db_index=True)
    state = models.TextField(default='pending', db_index=True) # pending / done
    line_userId = models.TextField(default='None', db_index=True)
    line_replyToken = models.TextField(default='None', db_index=True)
    line_timestamp = models.IntegerField(default=-1, db_index=True)
    line_type = models.TextField(default='None', db_index=True)
    line_event = models.TextField(default='None')
    
    def _get_line_text(self):
        try:
            evd = json.loads(self.line_event)
            return evd['message']['text']
        except KeyError:
            return None
    line_text = property(_get_line_text)
    @classmethod
    def from_event_dict(cls, event_dict: dict):
        lineID = event_dict['source']['userId']
        obj = cls(
            line_userId = lineID,
            line_replyToken = event_dict['replyToken'],
            line_timestamp = event_dict['timestamp'],
            line_type = event_dict['type'],
            line_event = json.dumps(event_dict),
        )
        if UserProfile.objects.filter(line_userId=lineID).count()==0:
            UserProfile(line_userId=lineID).save()
        return obj

def GetUserDialog(line_userId: str):
    r'''token=None means all done'''
    object_set = UserEvent.objects
    object_set = object_set.filter(line_userId=line_userId)
    object_set = object_set.filter(line_type='message')
    text_list = []
    ent_last_pending = None
    token = None
    for evt_object in object_set.order_by('created'):
        txt = evt_object.line_text
        if type(txt) != type('text msg'):
            continue
        if evt_object.state=='pending':
            ent_last_pending = evt_object
        if evt_object.state=='done':
            # first get 'done' msg only
            text_list.append(txt)
            
            # valid pending should after the "done" one
            ent_last_pending = None
    if type(ent_last_pending) != type(None):
        text_list.append(ent_last_pending.line_text)
        token = ent_last_pending.line_replyToken
        
        # record last pending pk
        user = UserProfile.objects.get(line_userId=line_userId)
        user.pk_last_pending = ent_last_pending.id
        user.save()
        
    return text_list, token

def GetTokenUpdateUserDone(line_userId: str):
    user = UserProfile.objects.get(line_userId=line_userId)
    event = UserEvent.objects.get(pk=user.pk_last_pending)
    event.state = 'done'
    event.save()
    return event.line_replyToken