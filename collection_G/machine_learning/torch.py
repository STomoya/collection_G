"""
Functions around machine learning

for pytorch
"""

try:
    import torch
except:
    torch = None

from ._exception import *

def torch_fit(

):
    
    if not torch:
        raise_no_module_error("torch")
    
