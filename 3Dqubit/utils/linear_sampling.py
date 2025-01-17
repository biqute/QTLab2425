import numpy as np

def linear_sampling(samplesx, datax, datay):
    samplesy = samplesx * 0
    for i, x in enumerate(samplesx):
        just_before_index = np.argmax(np.where(datax < x, datax, float("-inf")))
        just_after_index = np.argmin(np.where(datax > x, datax, float("inf")))
        
        if x in datax:
            samplesy[i] = datay[np.argmax(datax == x)]
        elif x <= np.min(datax):
            samplesy[i] = datay[0]
        elif x >= np.max(datax):
            samplesy[i] = datay[-1]
        else:
            slope = (datay[just_after_index] - datay[just_before_index]) / (datax[just_after_index] - datax[just_before_index])
            samplesy[i] = datay[just_before_index] + slope * (x - datax[just_before_index])

    return samplesy