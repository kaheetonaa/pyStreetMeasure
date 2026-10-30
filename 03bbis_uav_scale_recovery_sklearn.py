# (s,t)= argmin s,t SUM(s*d_pred + t - d_ref)**2

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import cv2
import math

pred=np.load('LC3D/pred_0092.npy')
true=np.load('LC3D/height-ref.npy')
mask=np.load("LC3D/mask-0092.npy")

pred=cv2.resize(pred,(8192,5460),interpolation=cv2.INTER_NEAREST)
pred=(pred-pred.min())/(pred.max()-pred.min())

RADIUS=10
MASK=0

_pred,_true=[],[]
#plt.imshow(mask)
#plt.show()
def cal_dist(_a:list,_b:list):
    return math.sqrt((_a[0]-_b[0])**2+(_a[1]-_b[1])**2)/math.sqrt((_b[0])**2+(_b[1])**2)

#true would have shape [x,y,depth]
for i in range(true.shape[0]):
    _x,_y=int(true[i][0]),int(true[i][1])
    if (mask[_y,_x]>0 and MASK==1):
        break
    else:
    #_pred += [pred[_y+_j, _x+_k] for _j in range(-2, 3) for _k in range(-2, 3)]
        _pred +=[np.mean(pred[(_y-RADIUS):(_y+RADIUS),( _x-RADIUS):(_x+RADIUS)])]#(_x-8192/2)/8192,(_y-5460/2)/5460
        _true+=[true[i][2]]
print(pred.shape,pred.min(),pred.max()) #y,x


_pred,_true=np.array(_pred).reshape(-1,1),np.array(_true)
print(np.isnan(_true.any()))
plt.scatter(_pred,_true)
plt.show()
pred_train,pred_test,true_train,true_test=train_test_split(
_pred, _true, test_size=30, random_state=42)

regressor = LinearRegression().fit(pred_train, true_train)
val = regressor.predict(pred_test)

print(f"Mean squared error: {mean_squared_error(val, true_test):.2f}")
print(f"Coefficient of determination: {r2_score(val, true_test):.2f}")



rows, cols = np.indices((pred.shape[0], pred.shape[1]))



#pred=np.stack([rows,cols, pred], axis=-1).astype(np.float32)
rescale=regressor.predict((pred).reshape(-1,1))
print(rescale.shape)
rescale=rescale.reshape(pred.shape[0],pred.shape[1])
plt.imshow(rescale)
plt.show()
np.save('LC3D/rescale-c.npy',rescale)
