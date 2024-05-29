# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:55:48 2020

@author: Hanbz
"""

import cffi
ffibuilder = cffi.FFI()

header="""
extern void pymodel(double *, int *, int *, double *, int *, int *, double *, int *, int *);
"""

module = """
import my_module
import copy
from my_plugin import ffi

@ffi.def_extern()
def pymodel(batch_xs, shape1_x, shape2_x,step,shape1_step, shape2_step,batch_ys,shape1_y, shape2_y):
    import numpy as np
    import torch
    from myCNN_torch import CNN
    
    pys=my_module.asarray(ffi, batch_xs, shape1_x, shape2_x)
    pystep=my_module.asarray(ffi, step, shape1_step, shape2_step)
    pya=my_module.asarray(ffi, batch_ys, shape1_y, shape2_y)
        
    if pystep[0][0]==0.0:
        global loadCNN
        loadCNN= CNN([1, 6, 6, 6, 1], [2, 2])
        loadCNN.initialize_layers()
        loadCNN.double()
        with open('model.pkl', 'rb') as f:
            loadCNN.load_state_dict(torch.load(f, map_location=torch.device('cpu')))

        print(loadCNN)
        print("Initialized")

    else:
        input = torch.tensor(pys[np.newaxis,np.newaxis,:,:])
        a = loadCNN(input.double()).detach().numpy().squeeze()

        #call back data
        pya[:,:]=a[:,:]
    
"""
#with open("plugin.h","w") as f:
#    f.write(header)
    
ffibuilder.embedding_api(header)
ffibuilder.set_source("my_plugin", "")

ffibuilder.embedding_init_code(module)
ffibuilder.compile(target="plugin.*", verbose=True)