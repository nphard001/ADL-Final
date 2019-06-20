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


