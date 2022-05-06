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