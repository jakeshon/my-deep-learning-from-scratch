import numpy as np
import common.functions as cf

def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
x = cf.gradient_descent(function_2, init_x, lr=0.1, step_num=100 )
