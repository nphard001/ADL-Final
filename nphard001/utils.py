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