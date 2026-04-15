import numpy as np
#import pandas as pd
import viser
import viser.extras.colmap as colmap

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error


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

def lasso_loss(X, y,predictions, beta, lam):
    n = len(y)
    residuals = y - predictions
    loss = (1 / (2 * n)) * np.sum(residuals ** 2) + lam * np.sum(np.abs(beta))
    return loss


def Polyfit(x,y,degree):
    coeffs = np.polyfit(x, y, degree)   # returns coefficients [a_n, ..., a_1, a_0]
    poly   = np.poly1d(coeffs)          # wraps coeffs into a callable polynomial
    return poly


def Polyfit_check_degree(x,y):
    for deg in range(1, 7):
        c = np.polyfit(x, y, deg)
        pred=np.poly1d(c)(x)
        res = y - pred
        r2 = 1 - np.sum(res**2) / np.sum((y - y.mean())**2)
        lambda_loss=lasso_loss(x,y,pred,c,lam=0.5)
        print(f"deg {deg}: R²={r2:.4f}")
        print(f"deg {deg}: lambdaloss={lambda_loss:.4f}")


def MDE_filter(mde_input,filter_array):
    #input: mde row:y column:x att:depth; filter_array:[x,y]
    mde_df=[np.array([0,0,0])]
    for j in filter_array:
        mde_df=np.r_[mde_df,[[int(j[0]),int(j[1]),mde_input[int(j[1]),int(j[0])]]]]
    mde_df=mde_df[1:]
    return mde_df


dense_array=np.array([[0,0]])
for i in range(0,mde_input.shape[1],50):
    for j in range(0,mde_input.shape[0],50):
        dense_array=np.r_[dense_array,np.array([[i,j]])]
dense_array=dense_array[1:]

print(dense_array)

Polyfit_check_degree(MDE_filter(mde_input,df)[:,2],df[:,6])
polyscale=Polyfit(MDE_filter(mde_input,df)[:,2],df[:,6],3)

# +
after_2d=MDE_filter(mde_input,df)
after_2d[:,2]=polyscale(after_2d[:,2])

dense_2d=MDE_filter(mde_input,dense_array)
dense_2d[:,2]=polyscale(dense_2d[:,2])

dense_2d=dense_2d[dense_2d[:,2]>0]
dense_2d=dense_2d[dense_2d[:,2]<80]

before=twoD_to_cam_reproject(col_intrinsic,params,images[1],MDE_filter(mde_input,df))
after=twoD_to_cam_reproject(col_intrinsic,params,images[1],after_2d)
dense=twoD_to_cam_reproject(mde_intrinsic,params,images[1],dense_2d)

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

