#K=3x3 matrix
#extrinsic=[R|t] 4x3 matrix
#cam_coord in ecef-->enu raster local
#raster=wxh

#import libs
import rasterio
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import pandas as pd
#step 0 all attribute


img_omega=-0.0001474591126878891
img_phi=0.07188948967356248# +180 because the camera face down
img_kappa=90.11738322173593-90# +90 because of rotation
img_w=8192
img_h=5460
img_fx=5460.92311
img_fy=5460.92311
img_lat=45.96345536111111
img_lon=7.917383638888889
img_alt=2142.469

dataset = rasterio.open('LC3D/w50540_s10_ellipsoid.tif')#tinyitaly data
def convert_projection(_coord_in,_epsg_in:str,_epsg_out:str):
        #_coord_in: (latitude,longitude)
        #_epsg_in: input projection (e.g. "EPSG:xxxx")
        #_epsg_out: output projection (e.g. "EPSG:xxxx")
        _t=Transformer.from_crs(_epsg_in,_epsg_out)
        _coord_out=_t.transform(_coord_in[0],_coord_in[1])
        return _coord_out

img_pos=convert_projection((img_lat,img_lon),"EPSG:4326",dataset.crs)

img_eul_ang=[img_omega,img_phi,img_kappa]
Rot = R.from_euler("xyz", img_eul_ang, degrees=True).as_matrix()
trans = [img_pos[0],img_pos[1],img_alt]
K_intr = np.array([[img_fx,0,img_w/2],
                   [0,img_fy,img_h/2],
                   [0,0,1]])
inv_z=np.array([[1,0,0],[0,-1,0],[0,0,-1]])

#step 1 load npy image
depth=np.load("scale0080.npy")
print(depth)
plt.imshow(depth)
plt.show()
print(depth.shape)
#step 2 column row to u v
pos=[]
for _v in range(0,depth.shape[0],50):
    for _u in range(0,depth.shape[1],50):
        _uv=np.array([_u,_v,depth[_v,_u]])
        _uv[:2]=_uv[:2]*_uv[2]
        pos+=[np.linalg.inv(Rot)@(np.linalg.inv(inv_z)@(np.linalg.inv(K_intr)@_uv))+trans]
        #print(_u,_v)
pos=np.array(pos)

with np.printoptions(suppress=True, precision=3):
    df=pd.DataFrame(pos,columns=["x","y","z"])
    df.to_csv("LC3D/mde_dem_0080.csv")
