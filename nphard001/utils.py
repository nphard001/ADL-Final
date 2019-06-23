# utils.py: common functions (copy from package newremote)
import os, sys, json
from typing import Callable, List, Dict, Optional, ClassVar
from collections import OrderedDict, defaultdict
import datetime
import hashlib
now = lambda: datetime.datetime.now().timestamp()
md5 = lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()

def UniqueList(lst: List)->List:
    return list(OrderedDict.fromkeys(lst))

class CallableDict(dict):
    r'''base class kind of manager, easy to log'''
    def __init__(self, fn=None, dict_init=None):
        self._fn = fn or self.get_default_fn()
        if dict_init:
            self.update(dict_init)
    def __call__(self, *args):
        return self._fn(*args)
    def get_default_fn(self):
        fn = lambda *args: print(*args, file=sys.stderr)
        return fn

from io import StringIO
def print2str(*args, **kwargs):
    buf = StringIO()
    print(*args, **kwargs, file=buf)
    return buf.getvalue()

import numbers
import numpy as np
def _ForceNumber(x):
    return x if isinstance(x, numbers.Number) else float('nan')
def _ForceString(x):
    return x if type(x)==type('') else repr(x)
def _Key(x):
    return repr(x)
def LOLPosition(*lol):
    list_flatten = []
    list_to_val = []
    pos_table = {}
    for lst in lol:
        list_flatten.extend(lst)
    xnum = [_ForceNumber(x) for x in list_flatten]
    xstr = [_ForceString(x) for x in list_flatten]
    list_to_val = (np.lexsort([xstr, xnum]).argsort() / len(xnum))
    for idx, obj in enumerate(list_flatten):
        if _Key(obj) not in pos_table:
            pos_table[_Key(obj)] = list_to_val[idx]
    lol_ans = []
    for lst in lol:
        lol_ans.append([pos_table[_Key(x)] for x in lst])
    return lol_ans
# LOLPosition([7, 3, 3, 1, 3.0, 'GRU', None], [None, None, 'GRU', 3.5])
