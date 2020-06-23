

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
import numpy as np

dm = [0,5,5,6,3,6]
md = [0,111,112,126,70,126]

dmn = np.array(dm)
mdn = np.array(md)
dmn
mdn
nbr_loop = int(max(np.where(np.isnan(np.divide(mdn,dmn)),0,np.ceil(np.divide(mdn,dmn)))))
nbr_loop = int(max(np.where(np.isnan(np.divide(mdn,dmn)),0,np.ceil(np.divide(mdn,dmn)))))
for i in range(nbr_loop):
    print(f'remaining demand after {i+1} runs')
    print(np.where(mdn-(i+1)*dmn < 0,0,mdn-(i+1)*dmn), '\n')
