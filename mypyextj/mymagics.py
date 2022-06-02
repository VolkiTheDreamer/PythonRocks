from IPython.core.magic import register_line_magic, register_cell_magic,register_line_cell_magic
import warnings
import os, sys, site
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell,display
from IPython.display import HTML
from IPython.utils.capture import capture_output
from IPython.display import Audio
from playsound import playsound
import winsound
import inspect
# from __future__ import print_function
import inspect
import functools
import time
import subprocess

# try:
#     import __builtin__
# except ImportError:
#     import builtins as __builtin__

#1-this file is executed on startup as it is in C:\Users\username\.ipython\profile_default\startup

#2-all this code is executed thanks to first bullet, so no need to run them manually each time.
shell=InteractiveShell()
shell.run_cell("""
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'""")

shell.run_cell("""
%load_ext autoreload
%autoreload 2
""")

#3-tip
"""
If you want to have some script imported conditionally, you could insert the foloowing code inside an if block
if foo==bar:
    shell.run_cell(raw_cell='from .... import.....')
"""
    
@register_line_magic        
def mgc_suppresswarning(line):
    warnings.filterwarnings("ignore",category=eval(line))

@register_line_magic  
def mgc_pythonSomeInfo(line):    
    print("system packages folder:",sys.prefix, end="\n\n")
    print("pip install folder:",site.getsitepackages(), end="\n\n")    
    print("python version:", sys.version, end="\n\n")
    print("executables location:",sys.executable, end="\n\n")
    print("pip version:", os.popen('pip version').read(), end="\n\n")
    pathes= sys.path
    print("Python pathes")
    for p in pathes:
        print(p)        
        
@register_cell_magic
def mgc_beep(line, cell=None):   
    exec_val = line if cell is None else cell
    with capture_output(True, False, True) as io:
        shell.run_cell(raw_cell=exec_val)    
    io.show()
#     playsound(r"C:\Windows\Media\notify.wav")
    winsound.PlaySound(r"C:\Windows\Media\\notify.wav", winsound.SND_FILENAME)
    # winsound.Beep(1000, 100)    

@register_line_magic  
def mgc_nbsnippet(line):
    """
        Converts a cell content,which was assigned to a string variable, into a nbextensions snippet-menu-compatible format.
    """
    temp=[]
    for x in eval(line).split("\n"):     
        rep=x.replace("\"","'")
        temp.append("\""+rep+"\"")

    final=",\n".join(temp)
    print(final)    


@register_line_magic  
def mgc_nbsnippet_revert(line):
    """
    Provide the string as json value, this will return th version without " signs.
    """
    temp=[x[1:-2] for x in eval(line).split("\n")]
    final="\n".join(temp)
    print(final)  

@register_line_magic    
def mgc_read_filecontent(line,enc='cp1254'):
    with open(line, "r", encoding=enc) as f:
        return f.read()

@register_line_magic  
def mgc_isLatestVersion(line):
    """
        Provides the packagename without quotation marks and import the package in advance
    """   
    packagename=line
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
    



#*****************************NON-MAGICS************************************
#THESE WILL BE READY AS SOON AS JUPYTER LAUNCHES AS THIS FILE IS IN STARTUP PATH
# 
# def printy(*args, **kwargs):
#     """
#         prints the results along with the line number in a cell containing  multiple print statement to prevent confusions.
#     """
#     def lineno():
#         previous_frame = inspect.currentframe().f_back.f_back
#         (filename, line_number, function_name, lines, index) = inspect.getframeinfo(previous_frame)
#         return (line_number, lines[0][:-1])

#     print(lineno(), "\n----------")
#     print(*args, **kwargs)
#     print(" ",end="\n")


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
    
