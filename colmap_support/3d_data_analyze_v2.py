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
    #'x2d','y2d','xreproj','yreproj','z'
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
    #"x2d",y2d,'x3d','y3d','z'
    f,cx,cy,k=params
    ROT=np.array(image.qvec2rotmat())
    T=np.array(image.tvec)
    print(T)
    p3D_cam = np.linalg.inv(intrinsic) @ depth.T  # normalize
    p3D=np.c_[depth[:,:2],p3D_cam.T]
    return p3D

df=world_to_cam_reproject(params,images[1])
print(df)
print("----------------Training--------------------")


import numpy as np


def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].
    
    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    print(c,R,t)
    return c, R, t

def affine_fit(src, tgt):
    """
    src, tgt: (N, 3) known correspondences
    returns A (3x3), t (3,)
    such that (A @ src[i]) + t ≈ tgt[i]
    """
    N = len(src)

    # design matrix: [x, y, z, 1] per point  →  shape (N, 4)
    P = np.hstack([src, np.ones((N, 1))])

    # solve P @ M = tgt  for M (4x3)  — one lstsq call for all 3 dims
    M, residuals, rank, sv = np.linalg.lstsq(P, tgt, rcond=None)

    A = M[:3].T    # (3, 3)
    t = M[3]       # (3,)

    return A, t

def LeastSquared(X,y,title):
    print(X.shape,y.shape)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,shuffle=False)
    regressor = LinearRegression().fit(X_train, y_train)
    weight= regressor.coef_
    bias=regressor.intercept_
    print('done training ',title,'weight', weight,'bias',bias)
    y_pred = regressor.predict(X_test)

    print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"Coefficient of determination: {r2_score(y_test, y_pred):.2f}")
    return [weight,bias]

def MDE_filter(mde_input,filter_array):
    #input: mde row:y column:x att:depth; filter_array:[x,y]
    mde_df=[np.array([0,0,0])]
    for j in filter_array:
        mde_df=np.r_[mde_df,[[int(j[0]),int(j[1]),mde_input[int(j[1]),int(j[0])]]]]
    mde_df=mde_df[1:]
    for i in mde_df:
            i[0]=i[0]*i[2]
            i[1]=i[1]*i[2]
    return mde_df

dense_array=np.array([[0,0]])
for i in range(0,mde_input.shape[1],50):
    for j in range(0,mde_input.shape[0],50):
        dense_array=np.r_[dense_array,np.array([[i,j]])]
dense_array=dense_array[1:]


before=twoD_to_cam_reproject(mde_intrinsic,params,images[1],MDE_filter(mde_input,df))
after=twoD_to_cam_reproject(mde_intrinsic,params,images[1],MDE_filter(mde_input,df))
dense=twoD_to_cam_reproject(mde_intrinsic,params,images[1],MDE_filter(mde_input,dense_array))
print(before[:,2:].shape)
s,R,t=umeyama(before[:,2:].T,df[:,2:].T)
#A,t_rs,_=ransac_affine(before[:,2:],df[:,2:])
#after[:,2:]=(A@after[:,2:].T).T+t_rs
#dense[:,2:]=(A@dense[:,2:].T).T+t_rs
after[:,2:]=(s*(R@after[:,2:].T)+t).T
dense[:,2:]=(s*(R@dense[:,2:].T)+t).T
rmse = np.sqrt(np.mean(np.linalg.norm(after - before, axis=1)**2))
print("rmse=",rmse)
#x_scale=LeastSquared(before[:,[0]],df[:,2],"x")
#y_scale=LeastSquared(before[:,[1]],df[:,3],"y")
#z_scale=LeastSquared(before[:,[2]],df[:,4],"z")
#after[:,0]=after[:,0]*x_scale[0]+x_scale[1]
#after[:,1]=after[:,1]*y_scale[0]+y_scale[1]
#after[:,2]=after[:,2]*z_scale[0]+z_scale[1]

#dense[:,0]=dense[:,0]*x_scale[0]+x_scale[1]
#dense[:,1]=dense[:,1]*y_scale[0]+y_scale[1]
#dense[:,2]=dense[:,2]*z_scale[0]+z_scale[1]

print("----------------Visualization--------------------")

#print(dense)

def main():
    server = viser.ViserServer()
    server.scene.add_point_cloud(
            name="sparse",
            points=np.array(points3D_coord),
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

    while True:
        pass

if __name__=="__main__":
    main()

