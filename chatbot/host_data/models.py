import pickle
from nphard001.img import *
from nphard001.django_model import *
from urllib.parse import quote


class AttrMetadata(BaseModel):
    category = models.TextField(db_index=True, default='unknown')
    img_id = models.IntegerField(db_index=True, default=-1)
    img_path = models.TextField(default='None')
    dsr_path = models.TextField(default='None')
    @staticmethod
    def GetImgBinary(img_type:str='raw', ctg: str='earrings_drop', img_id: int=1694):
        obj = AttrMetadata.objects.filter(category=ctg, img_id=img_id)[0]
        if img_type=='raw':
            with open(obj.img_path, 'rb') as f:
                return f.read()
        elif img_type=='thumbnail':
            # image via Pillow
            img = Image.open(obj.img_path)
            img.thumbnail([120, 120]) # it's inplace
            img_bytes = BytesIO()
            img.save(img_bytes, 'jpeg')
            return img_bytes.getvalue()

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

class UserReplyImage(BaseModel):
    created = models.DateTimeField(auto_now_add=True, db_index=True)
    line_userId = models.TextField(default='None', db_index=True)
    train_idx = models.IntegerField(default=-1, db_index=True)
    info = models.TextField(default='None')

class AttrTrain(BaseModel):
    r'''some cacheable info related to train_im index (like pretrain features)'''
    tag = models.TextField(default='toy', db_index=True)
    idx = models.IntegerField(default=-1, db_index=True)
    info = models.TextField(default='None')
    @property
    def info_json(self):
        return json.loads(self.info)
    def complete(self):
        r'''derive info from (tag, idx) by an in-memory way'''
        self.info = json.dumps({'idx': self.idx, 'value': self.idx**2})
        return self
    @classmethod
    def get_cache(cls, **query_dict):
        try:
            return cls.objects.get(**query_dict)
        except cls.DoesNotExist:
            obj = cls(**query_dict).complete()
            obj.save()
            return obj
    @classmethod
    def get_attr_list(cls, idx_list, **query_dict):
        return [cls.get_cache(idx=idx, **query_dict).info_json for idx in idx_list]
    @classmethod
    def pack_256embedding(cls, src: str='../static/model/features/256embedding.p', tag: str='256embedding', N: int=10000):
        if cls.objects.filter(tag=tag).count() == N:
            print('[pack_256embedding] already done', file=sys.stderr)
            return
        list_of_np = pickle.load(open(src, 'rb'))['train']
        for idx in range(N):
            print(f'[pack_256embedding] idx={idx}', file=sys.stderr)
            info = json.dumps([float(f) for f in list_of_np[idx]])
            cls.objects.create(tag=tag, idx=idx, info=info)

# ================================================================
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

def GetTrainURLByIndex(train_idx: int):
    def _wrap(linux7):
        heroku = 'https://nphard001.herokuapp.com/line/identity?url='
        return heroku+quote(linux7, safe="")
    url_240x240 = _wrap(f'https://linux7.csie.org:3721/data/image/thumbnail/train/{train_idx}')
    url = _wrap(f'https://linux7.csie.org:3721/data/image/raw/train/{train_idx}')
    return url_240x240, url

def GetGridResnet2d(id_list, nrow: int, ncol: int):
    # fetch features
    res256 = AttrTrain.get_attr_list(id_list, tag='256embedding')
    res256_pca = AttrTrain.get_attr_list(
        np.random.RandomState(1337).choice(10000, 3333, False), tag='256embedding')
    assert len(res256[0])==256
    X = np.array(res256)
    X_pca = np.array(res256_pca)
    from sklearn.decomposition import PCA
    pca = PCA(2)
    pca.fit(X_pca)
    def _pca_val(vec, i_dim):
        vec2 = pca.transform(vec.reshape(1, -1)).reshape(-1)
        return float(vec2[i_dim])
    val1 = LOLPosition([_pca_val(ary256, 0) for ary256 in X])[0]
    val2 = LOLPosition([_pca_val(ary256, 1) for ary256 in X])[0]
    
    def _find_img_id(qval1, qval2):
        best = id_list[0]
        best_dist = 9999.99
        for i, img_id in enumerate(id_list):
            dist = abs(val1[i]-qval1) + abs(val2[i]-qval2)
            if best_dist > dist:
                best = img_id
                best_dist = dist
        return best
    
    # fill in grid
    _i2f = lambda i, n: (1.0/n)*(i+0.5)
    grid_2d = [[{} for _ in range(ncol)] for _ in range(nrow)]
    for i in range(nrow):
        row_val = _i2f(i, nrow)
        for j in range(ncol):
            col_val = _i2f(j, ncol)
            img_id = _find_img_id(row_val, col_val)
            grid_2d[i][j] = {
                'row_val': row_val,
                'col_val': col_val,
                'img_id': img_id,
                'title': print2str(i, j, round(row_val, 4), round(col_val, 4))
            }
    return grid_2d