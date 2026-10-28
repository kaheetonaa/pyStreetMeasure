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


img_omega=180-179.92986808165162
img_phi=-0.015833472534527618# +180 because the camera face down
img_kappa=167.27655197819288-90# +90 because of rotation
img_w=8192
img_h=5460
img_fx=5460.92311
img_fy=5460.92311
img_lat=45.96379205555556
img_lon=7.923466638888889
img_alt=2072.714

dataset = rasterio.open('LC3D/w50540_s10.tif')#tinyitaly data

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
trans = [-img_pos[0],-img_pos[1],-img_alt]
K_intr = np.array([[img_fx,0,img_w/2],
                   [0,img_fy,img_h/2],
                   [0,0,1]])



#step 1 load npy image
depth=np.load("LC3D/rescale-b.npy")
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
        pos+=[np.linalg.inv(Rot)@(np.linalg.inv(K_intr)@_uv)-trans]
        print(_u,_v)
pos=np.array(pos)

with np.printoptions(suppress=True, precision=3):
    df=pd.DataFrame(pos,columns=["x","y","z"])
    df.to_csv("LC3D/mde_dem.csv")

"""
 def _old():
    #step 1: check camera coord, raster coord, plot them
    img_name="DJI_20260730112036_0010.jpg" #summerschool data
    img_rel_alt=152.903
    print(dataset.crs)
    print(dataset.bounds)
    print(dataset.width,dataset.height)
    dataset_height=dataset.read(1)

    #reprojected img_position
    print(img_pos)
    print(img_pos[0],img_pos[1])
    print(dataset.index(img_pos[0],img_pos[1]))

    #step 2: reproject all point into camera

    
    #step 3: calculate depth
    #filter pixel in +-125m with 10m step
    depth_reference=np.array([[-9999,-9999,-9999]])
    for _x in np.arange(img_pos[0]-500,img_pos[0]+500,10):
        for _y in np.arange(img_pos[1]-500,img_pos[1]+500,10):
            _row,_col=dataset.index(_x,_y)
            if (dataset_height[_row,_col]>0):
                _pos=np.array([_x,_y,dataset_height[_row,_col]])
                uv=K_intr@(Rot@(_pos+trans))
                uv[:2]=uv[:2]/uv[2]
                uv=uv.reshape(1,3)
                depth_reference=np.append(depth_reference,uv,axis=0)

    depth_reference=depth_reference[1:]
    depth_reference=depth_reference[depth_reference[:,0]>0]
    depth_reference=depth_reference[depth_reference[:,0]<img_w]
    depth_reference=depth_reference[depth_reference[:,1]>0]
    depth_reference=depth_reference[depth_reference[:,1]<img_h]

    def normalize(_arr:np.array):
        return (_arr-_arr.min())/(_arr.max()-_arr.min())

    print(depth_reference)

    plt.scatter(depth_reference[:,0],depth_reference[:,1],c=normalize(depth_reference[:,2]),cmap="Greys")
    plt.show()

    #step 4: export to npy
    np.save("LC3D/height-ref.npy",depth_reference)

"""
