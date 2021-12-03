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


def get_shift_from_step(step, max_shift):
    quater = (step // max_shift) % 4 + 1
    if quater == 1:
        shift = step % max_shift
    elif quater == 2:
        shift = (step // max_shift + 2) % 2 * max_shift - step % max_shift
    elif quater == 3:
        shift = -1 * (step % max_shift)
    elif quater == 4:
        shift = -1 * ((step // max_shift + 2) % 2 * max_shift) + step % max_shift
    return shift
