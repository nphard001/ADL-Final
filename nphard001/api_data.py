from nphard001.api import *
from nphard001.attr_parser import AttrFilenameSplit

class HostDataAPI:
    r'''http wrapper, load attributedata from remote'''
    def __init__(self, host: str=r'https://linux7.csie.org:3721'):
        host = host.rstrip('/')
        self.host = host
        self._r = None
        self._j = None
    @property
    def last_response(self):
        return self._r
    @property
    def last_json(self):
        return self._j
    def get_metadata(self):
        return HTTPJson2Json(f'{self.host}/data/attributedata', {
            'type': 'metadata',
        })
    def get_txt(self, category: str='earrings_studs', img_id: int=1555):
        return HTTPJson2Json(f'{self.host}/data/attributedata', {
            'type': 'txt',
            'category': category,
            'img_id': img_id,
        })['target']
    def get_jpg(self, category: str='earrings_studs', img_id: int=1555):
        r'''get jpg and convert it to Pillow Image type'''
        return HTTPJson2Image(f'{self.host}/data/attributedata', {
            'type': 'jpg',
            'category': category,
            'img_id': img_id,
        })
    def get_jpg_by_name(self, im_name: str='img_womens_pumps_813.jpg'):
        return self.get_jpg(*AttrFilenameSplit(im_name))
    def get_txt_by_name(self, im_name: str='img_womens_pumps_813.jpg'):
        return self.get_txt(*AttrFilenameSplit(im_name))
        
def get_metadata_static_dict():
    r'''
    static metadata, example:
    metadata["bags_backpacks"]={"num": 226, "min_id": 1, "max_id": 226}
    '''
    return {
     "bags_backpacks": {
      "num": 226,
      "min_id": 1,
      "max_id": 226
     },
     "bags_clutch": {
      "num": 1643,
      "min_id": 1,
      "max_id": 1643
     },
     "bags_evening": {
      "num": 1681,
      "min_id": 1,
      "max_id": 1681
     },
     "bags_hobo": {
      "num": 1630,
      "min_id": 1,
      "max_id": 1630
     },
     "bags_shoulder": {
      "num": 1596,
      "min_id": 1,
      "max_id": 1596
     },
     "bags_totes": {
      "num": 1577,
      "min_id": 1,
      "max_id": 1577
     },
     "bags_wristlet": {
      "num": 792,
      "min_id": 1,
      "max_id": 792
     },
     "earrings_chandelier": {
      "num": 613,
      "min_id": 1,
      "max_id": 613
     },
     "earrings_diamond_studs": {
      "num": 1665,
      "min_id": 1,
      "max_id": 1665
     },
     "earrings_drop": {
      "num": 1703,
      "min_id": 1,
      "max_id": 1703
     },
     "earrings_hoops": {
      "num": 1750,
      "min_id": 1,
      "max_id": 1750
     },
     "earrings_pearl": {
      "num": 1721,
      "min_id": 1,
      "max_id": 1721
     },
     "earrings_studs": {
      "num": 1783,
      "min_id": 1,
      "max_id": 1783
     },
     "ties_bow": {
      "num": 1785,
      "min_id": 1,
      "max_id": 1785
     },
     "ties_plaid": {
      "num": 150,
      "min_id": 1,
      "max_id": 150
     },
     "ties_silk": {
      "num": 1798,
      "min_id": 1,
      "max_id": 1798
     },
     "ties_striped": {
      "num": 917,
      "min_id": 1,
      "max_id": 917
     },
     "womens_athletic_shoes": {
      "num": 1651,
      "min_id": 1,
      "max_id": 1651
     },
     "womens_boots": {
      "num": 1622,
      "min_id": 0,
      "max_id": 1621
     },
     "womens_clogs": {
      "num": 1649,
      "min_id": 1,
      "max_id": 1649
     },
     "womens_flats": {
      "num": 1519,
      "min_id": 1,
      "max_id": 1519
     },
     "womens_high_heels": {
      "num": 1670,
      "min_id": 1,
      "max_id": 1670
     },
     "womens_pumps": {
      "num": 1618,
      "min_id": 1,
      "max_id": 1618
     },
     "womens_rain_boots": {
      "num": 1037,
      "min_id": 1,
      "max_id": 1037
     },
     "womens_sneakers": {
      "num": 1157,
      "min_id": 1,
      "max_id": 1157
     },
     "womens_stiletto": {
      "num": 1700,
      "min_id": 1,
      "max_id": 1700
     },
     "womens_wedding_shoes": {
      "num": 1141,
      "min_id": 1,
      "max_id": 1141
     }
    }

def get_category_static_list():
    return [
    "bags_backpacks",
    "bags_clutch",
    "bags_evening",
    "bags_hobo",
    "bags_shoulder",
    "bags_totes",
    "bags_wristlet",
    "earrings_chandelier",
    "earrings_diamond_studs",
    "earrings_drop",
    "earrings_hoops",
    "earrings_pearl",
    "earrings_studs",
    "ties_bow",
    "ties_plaid",
    "ties_silk",
    "ties_striped",
    "womens_athletic_shoes",
    "womens_boots",
    "womens_clogs",
    "womens_flats",
    "womens_high_heels",
    "womens_pumps",
    "womens_rain_boots",
    "womens_sneakers",
    "womens_stiletto",
    "womens_wedding_shoes"
    ]