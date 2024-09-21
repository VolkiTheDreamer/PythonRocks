import time
import multiprocessing.pool as mpp

def getsquare(x):
    time.sleep(3)    
    return x*x

def get_power(x,power):
    time.sleep(3)    
    return x**power

def get_power_tpl(x):
    x,power=x
    time.sleep(3)    
    return x**power

def get_power_star(args):
    return get_power(*args)


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap
