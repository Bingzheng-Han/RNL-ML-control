# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# my_module.py
 
ctype2dtype = {}
import numpy as np 
#int变量
for prefix in ('int', 'uint'):
    for log_bytes in range(4):
        ctype = '%s%d_t' % (prefix, 8 * (2**log_bytes))
        dtype = '%s%d' % (prefix[0], 2**log_bytes)
#        print(ctype)
#        print(dtype)
        ctype2dtype[ctype] = np.dtype(dtype)

# float变量
ctype2dtype['float'] = np.dtype('f4')
ctype2dtype['double'] = np.dtype('f8')
ctype2dtype['int'] = np.dtype('i4')


def GetDataFromPointer(ffi, pointer):
    """
    将指针所指的变量值取出来
    """
    T = ffi.getctype(ffi.typeof(pointer).item)    # 获取变量的数据类型
    value = np.frombuffer(ffi.buffer(pointer, ffi.sizeof(T)),
                          ctype2dtype[T])    # 获得数据，此时得到的是一个数组
    value = value.item()    # 将数组中的元素取出来，只对包含单一元素的数组有效
    return value


def asarray(ffi, x_ptr, shape1_ptr, shape2_ptr):
    """
    x_ptr : Fortran传递过来的数组的地址
    shape1_ptr, shape2_ptr : Fortran数组的形状
    """
    shape = [GetDataFromPointer(ffi, pointer) for pointer in (shape1_ptr, 
             shape2_ptr)]
    length = np.prod(shape)    # 数组长度
    T = ffi.getctype(ffi.typeof(x_ptr).item)
    x = np.frombuffer(ffi.buffer(x_ptr, length * ffi.sizeof(T)),
                      ctype2dtype[T]).reshape(shape, order='F')
    return x