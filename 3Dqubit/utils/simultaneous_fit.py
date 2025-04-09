import numpy as np
import sys; sys.path.append("../classes")
from Fitter import Fitter

def simultaneous_fit(fitters):
    """Perform simultaneous fitting of multiple Fitter objects"""
    mfitter = Fitter()
    mfitter.datay = np.concat([f.datay for f in fitters])
    mfitter.sigmay = np.concat([f.sigmay for f in fitters])
    
    deltax = np.max([(np.max(f.datax) - np.min(f.datax)) for f in fitters]) + 1
    
    mfitter.datax = np.concat([(f.datax - np.min(f.datax) + i*deltax) for i, f in enumerate(fitters)])
    
    mfitter.params = dict()
    for f in fitters:
        for key, value in f.params.items(): 
            if key not in mfitter.params:
                mfitter.params[key] = value
                
    mfitter.derived_params = dict()
    for f in fitters:
        for key, value in f.derived_params.items(): 
            if key not in mfitter.derived_params:
                mfitter.derived_params[key] = value
    
    def model(datax, **params):
        # if type(datax) is not np.ndarray: datax = np.array([datax])
        datay = np.zeros(len(datax))
        for j, x in enumerate(datax):
            i = int(x / deltax)
            x = x + np.min(fitters[i].datax) - i*deltax
            iparams = dict()
            for key, value in params.items():
                if key in fitters[i].params:
                    iparams[key] = value
            
            datay[j] = fitters[i].model(x, **iparams)
        return datay
    
    par_names = ', '.join(mfitter.params.keys())
    par_names_and_values = ', '.join([f"{key}={key}" for key in mfitter.params.keys()])
    func_code = f'''def dynamic_func(x, {par_names}): return model(x, {par_names_and_values})'''
    local_namespace = {}
    exec(func_code, {'model': model}, local_namespace)
    named_model = local_namespace['dynamic_func']
        
    mfitter.model = named_model
    
    res = mfitter.fit()
    
    # TODO: create separate results for each fitter
    
    return res