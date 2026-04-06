import numpy as np
import pandas as pd
import viser
import viser.transforms as vtf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print('Done importing viser')
np.set_printoptions(precision=3)
path="model/"

print("------------Import MDE data------------------")
with open('depth.npy','rb') as f:
    mde_input=np.load(f)
print(mde_input.shape)

print("---------Import COLMAP data------------------")
cameras=read_cameras_binary(path+"sparse/0/cameras.bin")
images=read_images_binary(path+"sparse/0/images.bin")
points3D=read_points3d_binary(path+"sparse/0/points3D.bin")

print("--------Reading points coordinates----------")
#print(dir(images[1]))
points2D=images[1].xys[images[1].point3D_ids!=-1]
points3D_ids=images[1].point3D_ids[images[1].point3D_ids!=-1]
points3D_coord=[points3D[i].xyz for i in points3D_ids]
#print(points2D,points3D_ids)

print("----------------3D to 2D--------------------")
print(cameras[1].params)
params=cameras[1].params
def threeD_to_twoD_reproject(params,image):
    df=pd.DataFrame(columns=['x2d','y2d','xreproj','yreproj','z','depth'])
    intrinsic=np.array([[params[0],0,params[1]],
                    [0,params[0],params[2]],
                    [0,0,1]])
    ROT=np.array(image.qvec2rotmat())
    T=np.array(image.tvec)
    extrinsic=np.c_[ROT,T]
    f,cx,cy,k=params
    for i in range(len(points3D_coord)):
        p1=np.r_[np.array(points3D_coord[i]).T,np.array([1])]
        xreproj,yreproj,z=intrinsic.dot(extrinsic).dot(p1)
        if z>0:
            xreproj=(xreproj/z-cx)/f
            yreproj=(yreproj/z-cy)/f
            r2=xreproj**2 + yreproj**2
            factor= 1 + k*r2
            xreproj=float(cx + f*factor *xreproj)
            yreproj=float(cy + f*factor *yreproj)
            x2d,y2d=points2D[i]
            df.loc[len(df)]=[x2d,y2d,xreproj,yreproj,z,mde_input[int(y2d),int(x2d)]]
    return(df)
df=threeD_to_twoD_reproject(params,images[1])
print(df)

print("----------------Training--------------------")

X=df[['x2d','y2d','depth']]
y=df['z']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,shuffle=False)
regressor = LinearRegression().fit(X_train, y_train)
print('done training','weight', regressor.coef_,'bias',regressor.intercept_)

y_pred = regressor.predict(X_test)

print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coefficient of determination: {r2_score(y_test, y_pred):.2f}")
