"""Shared methods within all code"""
def tf_to_numpy(func):
    def inner(*args, **kwargs):
        res = func(*args, **kwargs)
        if isinstance(res, (list, tuple)):
            res = [item.numpy() for item in res]
        else:
            res = res.numpy()
        return res
    return inner  
