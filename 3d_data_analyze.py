import numpy as np
import viser
import viser.transforms as vtf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)
np.set_printoptions(precision=3)
path="model/"

cameras=read_cameras_binary(path+"sparse/0/cameras.bin")
images=read_images_binary(path+"sparse/0/images.bin")
points3D=read_points3d_binary(path+"sparse/0/points3D.bin")

print(dir(images[1]))
points2D=images[1].xys[images[1].point3D_ids!=-1]
points3D_ids=images[1].point3D_ids[images[1].point3D_ids!=-1]
print(points2D,points3D_ids)
print("-----------------------")

points3D_coord=[points3D[i].xyz for i in points3D_ids]
#print(points3D_coord)
print("-----------------------")
print(cameras[1].params)
params=cameras[1].params
intrinsic=np.array([np.zeros(3) for i in range(3)])
intrinsic[2,2]=1
intrinsic[0,0]=params[0]
intrinsic[1,1]=params[0]
intrinsic[0,2]=params[1]
intrinsic[1,2]=params[2]


print(intrinsic)
ROT=np.array(images[1].qvec2rotmat())
T=np.array(images[1].tvec)
extrinsic=np.c_[ROT,T]
p1=np.r_[np.array(points3D_coord[6]).T,np.array([1])]
print((intrinsic.dot(extrinsic).dot(p1)).astype(int))
print(points2D[6])

