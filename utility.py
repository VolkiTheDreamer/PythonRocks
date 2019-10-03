from __future__ import print_function
import inspect
import numpy as np

try:
    import __builtin__
except ImportError:
    import builtins as __builtin__
    
def lineno():
    return inspect.currentframe().f_back.f_back.f_lineno

def printy(*args, **kwargs):
    print(lineno(),"----------")
    print(*args, **kwargs)
    print(" ",end="\n")
    
