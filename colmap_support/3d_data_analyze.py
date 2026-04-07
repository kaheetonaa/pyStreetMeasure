import numpy as np
#import pandas as pd
import viser
import viser.extras.colmap as colmap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print('------------Done importing libraries--------')
np.set_printoptions(precision=3)
path="model/"

print("------------Import MDE data------------------")
with open('depth.npy','rb') as f:
    mde_input=np.load(f)
print(mde_input.shape)

print("---------Import COLMAP data------------------")
cameras=colmap.read_cameras_binary(path+"sparse/0/cameras.bin")
images=colmap.read_images_binary(path+"sparse/0/images.bin")
points3D=colmap.read_points3d_binary(path+"sparse/0/points3D.bin")

print("--------Reading points coordinates----------")
points2D=images[1].xys[images[1].point3D_ids!=-1]
points3D_ids=images[1].point3D_ids[images[1].point3D_ids!=-1]
points3D_coord=[points3D[i].xyz for i in points3D_ids]

print("----------------3D to 2D--------------------")
print(cameras[1].params)
params=cameras[1].params
intrinsic=np.array([[params[0],0,params[1]],
                    [0,params[0],params[2]],
                    [0,0,1]])
def threeD_to_twoD_reproject(params,image):
    #df=pd.DataFrame(columns=['x2d','y2d','xreproj','yreproj','deltax','deltay','z','depth'])
    df=[np.zeros(8)]
    f,cx,cy,k=params
    ROT=np.array(image.qvec2rotmat())
    T=np.array(image.tvec)
    extrinsic=np.c_[ROT,T]
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
            deltax=x2d-xreproj
            deltay=y2d-yreproj
            df=np.r_[df,[[x2d,y2d,xreproj,yreproj,deltax,deltay,z,mde_input[int(y2d),int(x2d)]]]]
    df=df[1:]
    return(df)

def twoD_to_threeD_reproject(params,image,predict):
    f,cx,cy,k=params
    ROT=np.array(image.qvec2rotmat())
    T=np.array(image.tvec)
    print(T)
    p3D=((np.linalg.inv(intrinsic)@np.linalg.inv(ROT))@predict.T).T#-T
    return p3D

df=threeD_to_twoD_reproject(params,images[1])
print(df)
print("----------------Training--------------------")

X=df[:,[0,1,7]]
y=df[:,6]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,shuffle=False)
regressor = LinearRegression().fit(X_train, y_train)
print('done training','weight', regressor.coef_,'bias',regressor.intercept_)
y_pred = regressor.predict(X_test)

print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coefficient of determination: {r2_score(y_test, y_pred):.2f}")

#mde_df=pd.DataFrame(columns=['x2d','y2d','depth'])
mde_df=[np.array([0,0,0])]
for c in range(0,len(mde_input),50):
    for r in range(0,len(mde_input[0]),50):
        mde_df=np.r_[mde_df,[[c,r,mde_input[c,r]]]]
mde_df=mde_df[1:]

mde_predict=regressor.predict(mde_df)

mde_df=mde_df[:,:2]

mde_df=np.c_[mde_df,mde_predict]

mde_df= mde_df[mde_df[:,2]>0]


for i in mde_df:
    i[0]=i[0]*i[2]
    i[1]=i[1]*i[2]

dense=twoD_to_threeD_reproject(params,images[1],mde_df)
print("----------------Visualization--------------------")

print(dense)

def main():
    server = viser.ViserServer()
    colors = np.zeros((dense.shape[0], 3), dtype=np.uint8)
    server.scene.add_point_cloud(
            name="dense",
            points=dense,
            colors=colors,
            point_size=1
            )
    while True:
        pass

if __name__=="__main__":
    main()

