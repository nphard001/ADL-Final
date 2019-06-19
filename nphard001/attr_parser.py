import os, re, json
def AttrFilenameSplit(f: str='img_womens_pumps_813.jpg'):
    r'''
    convert a row from "train_im_names.txt" format to (fctg, fid)
    example:
        input: img_womens_pumps_813.jpg
        output: ("womens_pumps", 813)
        '''
    return re.findall('img_([\S\W]+)_([\S\W]+).jpg', f)[0]
class AttrParser:
    def __init__(self, attr_root: str='../static/attributedata'):
        self.attr_root = attr_root
    def parse_metadata(self):
        list_of_dict = []
        for root, dirs, files in os.walk(self.attr_root):
            if len(files)>0:
                for f in files:
                    if 'img_' in f:
                        fctg, fid = AttrFilenameSplit(f)
                        f1 = os.path.join(root, f)
                        f2 = os.path.join(root, f'descr_{fctg}_{fid}.txt')
                        assert os.access(f1, os.R_OK)
                        assert os.access(f2, os.R_OK)
                        list_of_dict.append({
                            'img_id': int(fid),
                            'category': fctg,
                            'img_path': f1,
                            'dsr_path': f2,
                        })
        return list_of_dict
if __name__=='__main__':
    lod = AttrParser().parse_metadata()
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_main.settings')
    from django.core.wsgi import get_wsgi_application
    application = get_wsgi_application()
    from host_data.models import AttrMetadata
    AttrMetadata.objects.all().delete()
    cnt = 0
    for d in sorted(lod, key=lambda d: d['category'].__hash__()+d['img_id'] ):
        obj = AttrMetadata(
            img_id = d['img_id'],
            category = d['category'],
            img_path = d['img_path'],
            dsr_path = d['dsr_path']
        )
        obj.save()