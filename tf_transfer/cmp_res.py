from time import time

import numpy as np

from constants import *
from np_solver import calc as np_calc
from np_solver import pre_plot as np_pre_plot
from tf_solver import calc as tf_calc
from tf_solver import pre_plot as tf_pre_plot

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
        np_to_plot = np_pre_plot(np_res)
        start = performance(start)
        tf_res = tf_calc(F_tf)
        tf_to_plot = tf_pre_plot(tf_res)
        start = performance(start)
    
    total_res = [np_res, tf_res]
    # total_res = [np_to_plot[1], tf_to_plot[1]]
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
