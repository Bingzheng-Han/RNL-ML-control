# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:55:48 2020

@author: Hanbz
"""

import cffi
ffibuilder = cffi.FFI()

header="""
extern void pymodel(double *, int *, int *, double *, int *, int *, double *, int *, int *, double *, int *, int *, double *, int *, int *);
"""

module = """
import my_module
from myDDPG import myDDPG
import copy
import random
from my_plugin import ffi

@ffi.def_extern()
def pymodel(frs, shape1_s, shape2_s, fra, shape1_a, shape2_a, frr, shape1_r, shape2_r, frs_, shape1_s_, shape2_s_, frstep, shape1_step, shape2_step):
    import numpy as np
    import pickle

    pys=my_module.asarray(ffi, frs, shape1_s, shape2_s)
    pya=my_module.asarray(ffi, fra, shape1_a, shape2_a)
    pyr=my_module.asarray(ffi, frr, shape1_r, shape2_r)
    pys_=my_module.asarray(ffi, frs_, shape1_s_, shape2_s_)
    pystep=my_module.asarray(ffi, frstep, shape1_step, shape2_step)

    s=copy.deepcopy(pys)
    a=copy.deepcopy(pya)
    r=pyr[0][0]
    s_=copy.deepcopy(pys_)
    step=pystep[0][0]
    
    #print(step)
    if step==0.0:
        global ddpg,ep_reward,var,all_ep_r
        ddpg = myDDPG()
        ep_reward = 0
        all_ep_r = []
        var = 0.01  # control exploration
        print("----------Initialized----------")

        # Add exploration noise
        a_ = ddpg.choose_action(s_)
        a_ = np.random.normal(a_, var) # add randomness to action selection for exploration
        # a_ = np.clip(a_, -1, 1)
        #call back data
        pya[:,:]=a_[:,:]

    else:
        a_ = ddpg.choose_action(s_)
        a_ = np.random.normal(a_, var) # add randomness to action selection for exploration

        if int(step) % (ddpg.state_step) ==0:
            weights=ddpg.actor.state_dict()['layer1.0.weight'].numpy()
            r=r-abs(sum(np.squeeze(weights)))
            ddpg.replay_buffer.push((s, a, r, s_))
            ddpg.pointer += 1
            ep_reward += r

            if ddpg.pointer % (ddpg.MAX_EP_STEPS) == 0:
                print('Episode:', int(ddpg.pointer/ddpg.MAX_EP_STEPS), ' Reward: ', ep_reward, 'Explore: ', var)
                all_ep_r.append(ep_reward)
                np.savetxt('Moving_reward.plt',all_ep_r)
                ep_reward = 0.0
                ddpg.save(int(ddpg.pointer/ddpg.MAX_EP_STEPS))

            if ddpg.pointer >= ddpg.MEMORY_CAPACITY:
                # training
                var *= .9995    # decay the action randomness
                ddpg.learn()
                print('----------Updated----------')
        
        #call back data
        pya[:,:]=a_[:,:]
        pys[0,0:19]=np.squeeze(ddpg.actor.state_dict()['layer1.0.weight'].numpy())

"""
#with open("plugin.h","w") as f:
#    f.write(header)
    
ffibuilder.embedding_api(header)
ffibuilder.set_source("my_plugin", "")

ffibuilder.embedding_init_code(module)
ffibuilder.compile(target="plugin.*", verbose=True)