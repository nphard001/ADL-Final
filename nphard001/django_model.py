from nphard001.utils import *
from django.db import models
from django.utils import timezone
from django.forms.models import model_to_dict
class BaseModel(models.Model):
    class Meta:
        abstract = True
    def to_dict(self):
        return model_to_dict(self)
    def __str__(self):
        return json.dumps(model_to_dict(self), indent=1)
    def __repr__(self):
        return json.dumps(model_to_dict(self))

