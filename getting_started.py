#!/usr/bin/env python  
# coding: utf-8  


""" 
@version: v1.0 
@author: Styang 
@license: Apache Licence  
@contact: 460130107@qq.com 
@site:  
@software: PyCharm Community Edition 
@file: getting_started.py 
@time: 2019/3/11 21:22 
"""

from __future__ import print_function
import torch
import numpy as np

if __name__ == "__main__":

    x = torch.empty(5,3)
    print(x)

    x = torch.rand(5, 3)
    print(x)

    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)

    x = torch.tensor([5.5, 3])
    print(x)

    x = x.new_ones(5,3, dtype=torch.double)
    print(x)

    x = torch.randn_like(x, dtype=torch.float)
    print(x)

    print(x.size())

    y = torch.rand(5,3)

    print(x+y)
    print(torch.add(x,y))

    result = torch.empty(5,3)
    torch.add(x, y, out=result)
    print(result)

    # adds x to y, in-place
    # Any operation that mutates a tensor in-place is post-fixed with
    # an _. For example: x.copy_(y), x.t_(), will change x.
    y.add_(x)
    print(y)

    print(x)

    # standard NumPy-like indexing with all bells and whistles!
    print(x[:,1])


    # Resizing: If you want to resize/reshape tensor, you can use torch.view:
    x = torch.randn(4,4)
    print(x.size())

    y = x.view(size=(1,16))
    print(y.size())

    y = x.view(size=(-1,16))
    print(y.size())

    y = x.view(size=(1,-1))
    print(y.size())

    # the size -1 is inferred from other dimensions
    z = x.view((-1, 8))
    print(z.size())

    z = x.view((2, 2, -1))
    print(z.size())

    x = torch.randn([1])
    print(x)
    print(x.item())

    # If you have a one element tensor, use .item() to get the value as a Python number
    x = torch.randn(1)
    print(x)
    print(x.item())

    # Exception only one element tensors can be converted to Python scalars
    x = torch.randn(4,4)
    # print(x.item())


    #Converting a Torch Tensor to a NumPy array and vice versa is a breeze.

    #The Torch Tensor and NumPy array will share their underlying memory locations,
    # and changing one will change the other.

    a = torch.ones(5)
    print(a)

    b = a.numpy()
    print(b)

    # changing a will change the b
    a.add_(1)
    print(a)
    print(b)
    # however if you change b , a will not change  b=b+1 (not in-place)
    # a will change if doing as follow, because doing as this is in-place
    np.add(b, 1, out=b)
    print(a)
    print(b)



    a = np.ones(5)
    print(a)

    b = torch.from_numpy(a)
    print(b)

    np.add(a, 1, out=a)
    print(a)
    print(b)

    # let us run this cell only if CUDA is available
    # We will use ``torch.device`` objects to move tensors in and out of GPU

    if torch.cuda.is_available():
        print("true")
        # device = torch.device("cuda")
        device = torch.device("cuda")
        # directly create a tensor on GPU. or just use strings ``.to("cuda")``
        y = torch.ones_like(x, device=device)
        x = x.to(device)
        z = x + y
        print(z)

        # ``.to`` can also change dtype together!
        print(z.to("cpu", torch.double))
