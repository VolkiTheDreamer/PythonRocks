import time

def getsquare(x):
    time.sleep(5)    
    return x*x

def get_power(x,power):
    time.sleep(25)    
    return x**power

def get_power_tpl(x):
    x,power=x
    time.sleep(25)    
    return x**power

def merge_names(a, b):
    return '{} & {}'.format(a, b)