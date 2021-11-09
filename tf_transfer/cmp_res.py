from time import time

import numpy as np

from constants import *
from np_solver import calc as np_calc
from tf_solver import calc as tf_calc

def performance(start):
    end = time()
    print(end - start)
    return end

def main():
    F_np = init_poiseuille()
    F_tf = init_poiseuille()

    start = time()
    for step in range(5):
        print(f"step: {step}")
        np_res = np_calc(F_np)
        start = performance(start)
        tf_res = tf_calc(F_tf)
        start = performance(start)
    
    total_res = [np_res, tf_res]
    for name, item in zip(["np", "tf"], total_res):
        print(name, item.shape, item.mean())
    
    if isinstance(np_res, np.ndarray):
        assert np.all(np.isclose(np_res, tf_res))
    else:
        for np_item, tf_item in zip(np_res, tf_res):
            # assert np.all(np_item == tf_item)
            assert np.all(np.isclose(np_item, tf_item))
    # assert np.all(np_res == tf_res)


if __name__ == "__main__":
    main()
