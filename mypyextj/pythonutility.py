from __future__ import print_function
import inspect
import sys
import functools
import time
import subprocess
from IPython.display import Audio
from IPython.core.interactiveshell import InteractiveShell


try:
    import __builtin__
except ImportError:
    import builtins as __builtin__


def printy(*args, **kwargs):
    """
        prints the results along with the line number in a cell containing  multiple print statement to prevent confusions.
    """
    def lineno():
        previous_frame = inspect.currentframe().f_back.f_back
        (filename, line_number, function_name, lines, index) = inspect.getframeinfo(previous_frame)
        return (line_number, lines[0][:-1])

    print(lineno(), "\n----------")
    print(*args, **kwargs)
    print(" ",end="\n")


def timeElapse(func):
    """
        alternative to %time/timeit. In case of using an IDE apart from jupyter.
        
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

def showMemoryUsage():
    dict_={}
    global_vars = list(globals().items())
    for var, obj in global_vars:
        if not var.startswith('_'):
            dict_[var]=sys.getsizeof(obj)
            
    final={k: v for k, v in sorted(dict_.items(), key=lambda item: item[1],reverse=True)}    
    print(final)
    
def read_filecontent(path,enc='cp1254'):
    with open(path, "r", encoding=enc) as f:
        return f.read()


def isLatestVersion(packagename):
    """
        Provides the packagename without quotation marks and import the package in advance
    """   
    try:        
        name=str(packagename.__name__)    
        latest_version = str(subprocess.run([sys.executable, '-m', 'pip', 'install', '{}==random'.format(name)], capture_output=True, text=True))
        latest_version = latest_version[latest_version.find('(from versions:')+15:]
        latest_version = latest_version[:latest_version.find(')')]
        latest_version = latest_version.replace(' ','').split(',')[-1]

        current_version = str(subprocess.run([sys.executable, '-m', 'pip', 'show', '{}'.format(name)], capture_output=True, text=True))
        current_version = current_version[current_version.find('Version:')+8:]
        current_version = current_version[:current_version.find('\\n')].replace(' ','') 

        if latest_version == current_version:
            print(name+", "+packagename.__version__+", True")
        else:
            print(name+", "+packagename.__version__+", False")
    except NameError:
        print("You should import the package first")
    except:
        print("You should import the package first.")


