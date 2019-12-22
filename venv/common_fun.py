#!/usr/bin/python
from numpy import *
def txt_length(txt):
    return len(txt)
def txtAndlength(txt):
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    return len(txt),group


def len_fun(txt):
    if len(txt)<3:
        return txt+"<3"
    elif len(txt)>=3 and len(txt)<5:
        return 5
    else:
        return {"name":txt,"length":len(txt)}

aa,g=txtAndlength('uuuu')
print(aa,g)