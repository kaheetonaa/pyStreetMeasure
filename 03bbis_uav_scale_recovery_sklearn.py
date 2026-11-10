# (s,t)= argmin s,t SUM(s*d_pred + t - d_ref)**2

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import cv2
import math
import argparse

parser = argparse.ArgumentParser(prog="Scale Recovery",description ='rescale the MDE data using Least squared algorithm on near-ground-truth data',add_help=True)
parser.add_argument("--mde",help="MDE prediction npy file path",type=str,required=True)
parser.add_argument("--near_ground_truth",help="near-ground-truth data npy file path, in case UAV survey, it is reprojected ellipsodial DEM",type=str,required=True)
parser.add_argument("--mask",help="mask npy file path",type=str,default="NONE")
parser.add_argument("--radius",help="pixel manhattan distance around near-ground-truth pixel",type=int,default=1)
parser.add_argument("--output",help="output npy path",type=str,required=True)
args = parser.parse_args()
#--------------------------------------------------------------
#---load image-------------------------------------------------
pred=np.load(args.mde)
true=np.load(args.near_ground_truth)
MASK=0
if args.mask!="NONE":
    mask=np.load(args.mask)
    MASK=1
    plt.imshow(mask)
    plt.show()
RADIUS=args.radius

#--------------------------------------------------------------
#---resize image-------------------------------------------------
pred=cv2.resize(pred,(8192,5460),interpolation=cv2.INTER_CUBIC)
pred=(pred-pred.min())/(pred.max()-pred.min())



#--------------------------------------------------------------
#---prepare data-------------------------------------------------
_pred,_true=[],[]

def cal_dist(_a:list,_b:list):
    return math.sqrt((_a[0]-_b[0])**2+(_a[1]-_b[1])**2)/math.sqrt((_b[0])**2+(_b[1])**2)

#true would have shape [x,y,depth]
for i in range(true.shape[0]):
    _x,_y=int(true[i][0]),int(true[i][1])
    if (MASK==1 and mask[_y,_x]>0):
        continue
    if (np.isnan(np.mean(pred[(_y-RADIUS):(_y+RADIUS),( _x-RADIUS):(_x+RADIUS)]))):
        continue
    #_pred += [pred[_y+_j, _x+_k] for _j in range(-2, 3) for _k in range(-2, 3)]
    _pred +=[[np.mean(pred[(_y-RADIUS):(_y+RADIUS),( _x-RADIUS):(_x+RADIUS)])]]#(_x-8192/2)/8192,(_y-5460/2)/5460
    _true+=[true[i][2]]
print(pred.shape,pred.min(),pred.max()) #y,x
_pred,_true=np.array(_pred).reshape(-1,1),np.array(_true)
print(np.any(np.isnan(_true)))
print(np.all(np.isfinite(_true)))
print(np.any(np.isnan(_pred)))
print(np.all(np.isfinite(_pred)))
#plt.scatter(_pred,_true)
#plt.show()

#--------------------------------------------------------------
#---train and predict scale------------------------------------
for _random in range(0,100):
    pred_train,pred_test,true_train,true_test=train_test_split(
    _pred, _true, test_size=0.3, random_state=_random)

    #regressor = LinearRegression().fit(pred_train, true_train)
    regressor = HistGradientBoostingRegressor().fit(pred_train, true_train)
    val = regressor.predict(pred_test)

    print(f"Mean squared error: {mean_squared_error(val, true_test):.2f}")
    print(f"Coefficient of determination: {r2_score(val, true_test):.2f}")

#--------------------------------------------------------------
#---reshape and export model-----------------------------------
rows, cols = np.indices((pred.shape[0], pred.shape[1]))
print((rows/5460).max())


#pred=np.stack([pred,rows/5460,cols/8192], axis=-1).astype(np.float32)
rescale=regressor.predict((pred).reshape(-1,1))
print(rescale.shape)
rescale=rescale.reshape(pred.shape[0],pred.shape[1])
fig,ax=plt.subplots(2,1)
ax[0].imshow(pred)
ax[1].imshow(rescale)
plt.show()
np.save(args.output,rescale)
