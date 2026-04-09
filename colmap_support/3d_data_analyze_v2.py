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
with open('intrinsics.npy','rb') as f:
    mde_intrinsic=np.load(f)
with open('depth.npy','rb') as f:
    mde_input=np.load(f)
print(mde_input.shape,mde_intrinsic.shape)


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
col_intrinsic=np.array([[params[0],0,params[1]],
                    [0,params[0],params[2]],
                    [0,0,1]])
def world_to_cam_reproject(params,image):
    #df=pd.DataFrame(columns=['x2d','y2d','xreproj','yreproj','z'])
    df=[np.zeros(5)]
    f,cx,cy,k=params
    ROT=np.array(image.qvec2rotmat())
    T=np.array(image.tvec)
    for i in range(len(points3D_coord)):
        p1=np.array(points3D_coord[i]).T
        x2d,y2d=points2D[i]
        xreproj,yreproj,z=ROT@p1+T
        df=np.r_[df,[[x2d,y2d,xreproj,yreproj,z]]]
    df=df[1:]
    return(df)

def twoD_to_cam_reproject(intrinsic,params,image,depth):
    f,cx,cy,k=params
    ROT=np.array(image.qvec2rotmat())
    T=np.array(image.tvec)
    print(T)
    p3D_cam = np.linalg.inv(intrinsic) @ depth.T  # normalize
    return p3D_cam.T

df=world_to_cam_reproject(params,images[1])
print(df)
print("----------------Training--------------------")

#X=df[:,[7]]
#y=df[:,6]
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,shuffle=False)
#regressor = LinearRegression().fit(X_train, y_train)
#print('done training','weight', regressor.coef_,'bias',regressor.intercept_)
#y_pred = regressor.predict(X_test)

#print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
#print(f"Coefficient of determination: {r2_score(y_test, y_pred):.2f}")

mde_df=[np.array([0,0,0])]
for r in range(0,len(mde_input),25):
    for c in range(0,len(mde_input[0]),25):
        mde_df=np.r_[mde_df,[[c,r,mde_input[r,c]]]]
mde_df=mde_df[1:]
for i in mde_df:
        i[0]=i[0]*i[2]
        i[1]=i[1]*i[2]

before=twoD_to_cam_reproject(mde_intrinsic,params,images[1],mde_df)
print("----------------Visualization--------------------")

#print(dense)

def main():
    server = viser.ViserServer()
    server.scene.add_point_cloud(
            name="sparse",
            points=np.array(points3D_coord),
            colors=np.array([255,0,0]),
            point_size=.05
            )
    server.scene.add_point_cloud(
            name="non-finetuned",
            points=before,
            colors=np.array([0,0,255]),
            point_size=.05
            )

    while True:
        pass

if __name__=="__main__":
    main()

