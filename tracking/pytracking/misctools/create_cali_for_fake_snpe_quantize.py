import os
import numpy as np

INPUT_SHAPE1 = [1, 3, 112, 112]
INPUT_SHAPE2 = [1, 3, 240, 240]

basedir = os.path.split(__file__)[0]
calidir = os.path.join(basedir,'cali')
os.makedirs(calidir, exist_ok=True)
file_list = [(os.path.join(calidir,'z%06d.raw'%i),os.path.join(calidir,'x%06d.raw'%i)) for i in range(10)]
for z,x in file_list:
    np.random.randn(*INPUT_SHAPE1).astype(np.float32).tofile(z)
    np.random.randn(*INPUT_SHAPE2).astype(np.float32).tofile(x)
with open(os.path.join(calidir,'list1.txt'),'w') as f:
    #f.write('\n'.join(file_list))
    content = []
    for z,x in file_list:
        content.append(f'{z} {x}')
    f.write('\n'.join(content))
pass