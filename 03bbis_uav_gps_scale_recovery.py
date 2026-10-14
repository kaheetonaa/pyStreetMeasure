# (s,t)= argmin s,t SUM(s*d_pred + t - d_ref)**2

import numpy as np
import matplotlib.pyplot as plt
import cv2

pred=np.load('LC3D/pred_0092.npy')
true=np.load('LC3D/height-ref.npy')

#normalize pred
pred=(pred-pred.min())/(pred.max()-pred.min())

pred=cv2.resize(pred,(8192,5460),interpolation=cv2.INTER_CUBIC)
_pred,_true=[],[]

plt.imshow(pred)
plt.show()

#true would have shape [x,y,depth]
for i in range(true.shape[0]):
    _x,_y=int(true[i][0]),int(true[i][1])
    #_pred += [pred[_y+_j, _x+_k] for _j in range(-2, 3) for _k in range(-2, 3)]
    _pred += [pred[_y, _x]]
    _true+=[true[i][2]]

_true=np.array(_true)
_pred=np.array(_pred)
_pred=np.c_[_pred.T,np.ones(_pred.shape[0])]

scale_shift=np.linalg.solve(_pred,_true)

print(_pred,_true)
print(scale_shift)
pred_rescale=pred*scale_shift[0]+scale_shift[1]
np.save("LC3D/rescale.npy",pred_rescale)
