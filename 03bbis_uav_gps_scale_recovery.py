# (s,t)= argmin s,t SUM(s*d_pred + t - d_ref)**2

import numpy as np
import matplotlib.pyplot as plt
import cv2

pred=np.load('LC3D/pred_0092.npy')
true=np.load('LC3D/gps-height-ref.npy')

img_omega=-179.92986808165162
img_phi=-0.015833472534527618
img_kappa= 167.27655197819288
img_w=8192
img_h=5460
img_fx=5460.92311
img_fy=5460.92311
img_lat=45.96379205555556
img_lon=7.923466638888889
img_alt=2072.714

K_intr = np.array([[img_fx,0,img_w/2],
                   [0,img_fy,img_h/2],
                   [0,0,1]])


#normalize pred
pred=(pred-pred.min())/(pred.max()-pred.min())

pred=cv2.resize(pred,(8192,5460),interpolation=cv2.INTER_CUBIC)
_pred,_true=[],[]

select=np.array([[3103,1601,1],[2608,3519,1],[5472,1195,1],[6372,5020,1]]).T #select on the photo GCP



select_rescale=np.linalg.inv(K_intr)@select


for i in [0,3]:
    _x,_y=select_rescale[0][i],select_rescale[1][i]
    _u,_v=int(select[0][i]),int(select[1][i])
    print(_u,_v)
    _D=[pred[_v+j, _u+j] for j in range(-30,30)]
    _D=sum(_D)/len(_D) #average depth in range
    #_pred += [pred[_y+_j, _x+_k] for _j in range(-2, 3) for _k in range(-2, 3)]
    _pred += [_D]
    true[i][:2]=true[i][:2]*true[i][2]
    true[i][:3]=np.linalg.inv(K_intr)@true[i][:3]
    _true+=[true[i][:3]]

_true=np.array(_true).reshape(-1,1)
_pred=np.array(_pred).reshape(-1,1)
_pred=np.c_[_pred,np.ones(_pred.shape[0])]
_Q=np.identity(_true.shape[0])

print(_true)
print(_pred)
#quit()
scale_shift=_true
scale_shift=_pred.T@np.linalg.inv(_Q)@scale_shift
scale_shift=np.linalg.inv(_pred.T@np.linalg.inv(_Q)@_pred)@scale_shift
#TODO missing variance
print(scale_shift)
pred_rescale=pred*scale_shift[0]+scale_shift[1]
fig,ax=plt.subplots()
ax.scatter(true[:,0],true[:,1])
ax.imshow(pred_rescale)
plt.show()
np.save("LC3D/rescale_gps.npy",pred_rescale)
