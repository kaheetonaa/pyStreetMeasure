import numpy as np
#import pandas as pd
import viser
import viser.extras.colmap as colmap

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import  LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import matplotlib.pyplot as plt
import pandas as pd

print('------------Done importing libraries--------')
np.set_printoptions(precision=3)
path="model/"

print("------------Import MDE data------------------")
with open('intrinsics.npy','rb') as f:
    mde_intrinsic=np.load(f)
with open('depth.npy','rb') as f:
    mde_input=np.load(f)
print(mde_input.shape,"intrinsic: ",mde_intrinsic)


print("---------Import COLMAP data------------------")
cameras=colmap.read_cameras_binary(path+"sparse/0/cameras.bin")
images=colmap.read_images_binary(path+"sparse/0/images.bin")
points3D=colmap.read_points3d_binary(path+"sparse/0/points3D.bin")

print("--------Reading points coordinates----------")
points2D=images[1].xys[images[1].point3D_ids!=-1]
points3D_ids=images[1].point3D_ids[images[1].point3D_ids!=-1]
points3D_coord=[points3D[i].xyz for i in points3D_ids]


def threeD_to_twoD_reproject(intrinsic,params,image):
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


print("----------------3D to 2D--------------------")
print(cameras[1].params)
params=cameras[1].params
col_intrinsic=np.array([[params[0],0,params[1]],
                    [0,params[0],params[2]],
                    [0,0,1]])
def world_to_cam_reproject(params,image):
    #'x2d','y2d','xreproj','yreproj','z'
    df=[np.zeros(5)]
    f,cx,cy,k=params
    ROT=np.array(image.qvec2rotmat())
    T=np.array(image.tvec)
    for i in range(len(points3D_coord)):
        p1=np.array(points3D_coord[i]).T
        x2d,y2d=points2D[i]
        xcam,ycam,z=ROT@p1+T
        df=np.r_[df,[[x2d,y2d,xcam,ycam,z]]]
    df=df[1:]
    return(df)

def twoD_to_cam_reproject(intrinsic,params,image,d):
    #"x2d",y2d,'x3d','y3d','z'
    depth=d[:]
    for p in depth:
        p[0]*=p[2]
        p[1]*=p[2]
    f,cx,cy,k=params
    p3D_cam = np.linalg.inv(intrinsic) @ depth.T  # normalize
    p3D=np.c_[depth[:,:2],p3D_cam.T]
    return p3D

sparse=world_to_cam_reproject(params,images[1])[:,2:]
df=threeD_to_twoD_reproject(col_intrinsic,params,images[1])
print("----------------Training--------------------")

# +
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

def find_degree(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    param_grid = {'polynomialfeatures__degree': np.arange(10),
'linearregression__fit_intercept': [True, False]}
    grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("MSE:",mean_squared_error(y_test, y_pred))
    print("R2:",r2_score(y_test, y_pred))
    plt.scatter(x_test,y_pred)
    return model
       
def MDE_filter(mde_input,filter_array):
    #input: mde row:y column:x att:depth; filter_array:[x,y]
    mde_df=[np.array([0,0,0])]
    for j in filter_array:
        mde_df=np.r_[mde_df,[[int(j[0]),int(j[1]),mde_input[int(j[1]),int(j[0])]]]]
    mde_df=mde_df[1:]
    return mde_df

model=find_degree(MDE_filter(mde_input,df)[:,[2]],df[:,6])
model.get_params
#polyscale=Polyfit(MDE_filter(mde_input,df)[:,2],df[:,6],3)


dense_array=np.array([[0,0]])
for i in range(0,mde_input.shape[1],50):
    for j in range(0,mde_input.shape[0],50):
        dense_array=np.r_[dense_array,np.array([[i,j]])]
dense_array=dense_array[1:]


dense_2d=MDE_filter(mde_input,dense_array)
dense_2d[:,2]=model.predict(dense_2d[:,[2]])

after_2d=MDE_filter(mde_input,df)
after_2d[:,2]=model.predict(after_2d[:,[2]])

dense_2d=dense_2d[dense_2d[:,2]>0]
dense_2d=dense_2d[dense_2d[:,2]<80]

before=twoD_to_cam_reproject(col_intrinsic,params,images[1],MDE_filter(mde_input,df))
after=twoD_to_cam_reproject(col_intrinsic,params,images[1],after_2d)
dense=twoD_to_cam_reproject(col_intrinsic,params,images[1],dense_2d)


# +

print("----------------Visualization--------------------")

def main():
    server = viser.ViserServer()
    server.scene.add_point_cloud(
            name="sparse2",
            points=sparse,
            colors=np.array([255,0,0]),
            point_size=.5
            )
    server.scene.add_point_cloud(
            name="non-finetuned",
            points=before[:,2:],
            colors=np.array([0,0,255]),
            point_size=.1
            )
    server.scene.add_point_cloud(
            name="finetuned",
            points=after[:,2:],
            colors=np.array([0,255,0]),
            point_size=.1
            )
    server.scene.add_point_cloud(
            name="dense",
            points=dense[:,2:],
            colors=np.array([0,0,0]),
            point_size=.1
            )
    sphere = server.scene.add_icosphere(
        name="/sphere",
        radius=0.3,
        color=(255, 100, 100),
        position=(0.0, 0.0, 0.0),
    )
    while True:
        pass

if __name__=="__main__":
    main()

