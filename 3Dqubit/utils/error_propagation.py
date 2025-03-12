import math
import copy

def error_propagation(func, params):
    variance = 0
    original = func(params)
    for par in params:
        sx = params[par]["sigma"]
        dx = sx / 1e4
        new_params = copy.deepcopy(params)
        new_params[par]["value"] = params[par]["value"] + dx
        derivative = ( func(new_params) -  original) / dx
        variance += (derivative * sx)**2 

    return math.sqrt(variance)