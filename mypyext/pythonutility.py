from __future__ import print_function
import inspect
import os, sys, site
import functools
import time


try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

# *************************************************************************************************************
#Module level methods

    
def lineno():
    previous_frame = inspect.currentframe().f_back.f_back
    (filename, line_number, function_name, lines, index) = inspect.getframeinfo(previous_frame)
    return (line_number, lines)
    #return inspect.currentframe().f_back.f_back.f_lineno, str(inspect.currentframe().f_back)

def printy(*args, **kwargs):
    print(lineno(),"\n----------")
    print(*args, **kwargs)
    print(" ",end="\n")


def timeElapse(func):
    """
        usage:
        @timeElapse
        def somefunc():
            ...
            ...

        somefunc()
    """
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        start=time.time()
        value=func(*args,**kwargs)
        func()
        finito=time.time()
        print("Time elapsed:{}".format(finito-start))
        return value
    return wrapper    


def multioutput(type="all"):
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = type
    
def scriptforReload():
    print("""
    %load_ext autoreload
    %autoreload 2""")
   
def scriptforTraintest():
    print("X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)")
    
def scriptForCitation():
    print("""<p style="font-size:smaller;text-align:center">Görsel <a href="url">bu sayfadan</a> alınmıştır</p>""")
    
def pythonSomeInfo():
    print("system packages folder:",sys.prefix, end="\n\n")
    print("pip install folder:",site.getsitepackages(), end="\n\n")    
    print("python version:", sys.version, end="\n\n")
    print("executables location:",sys.executable, end="\n\n")
    print("pip version:", os.popen('pip version').read(), end="\n\n")
    pathes= sys.path
    print("Python pathes")
    for p in pathes:
        print(p)


def showMemoryUsage():
    dict_={}
    global_vars = list(globals().items())
    for var, obj in global_vars:
        if not var.startswith('_'):
            dict_[var]=sys.getsizeof(obj)
            
    final={k: v for k, v in sorted(dict_.items(), key=lambda item: item[1],reverse=True)}    
    print(final)
    
def readfile(path,enc='cp1254'):
    with io.open(path, "r", encoding=enc) as f:
        return f.read()

def getFirstItemFromDictionary(dict_):
    return next(iter(dict_)),next(iter(dict_.values()))
        







