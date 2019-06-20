from nphard001.django_model import *

class AttrMetadata(BaseModel):
    category = models.TextField(db_index=True, default='unknown')
    img_id = models.IntegerField(db_index=True, default=-1)
    img_path = models.TextField(default='None')
    dsr_path = models.TextField(default='None')

class UserEvent(BaseModel):
    created = models.DateTimeField(auto_now_add=True)
    line_userId = models.TextField(default='None')
    line_replyToken = models.TextField(default='None')
    line_event = models.TextField(default='None')