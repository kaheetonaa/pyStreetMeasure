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

#step 1: check camera coord, raster coord, plot them
img_name="DJI_20260730112036_0010.jpg"
img_lat=45.96045316666667
img_lon=7.919238027777778
img_omega=0.07007375125380845
img_phi=0.01607208000648593
img_kappa=12.919218966738043
img_w=8192
img_h=5460
img_fx=5460.92311
img_fy=5460.92311
img_alt=2143.776
img_rel_alt=152.903
dataset = rasterio.open('LC3D/w50540_s10.tif')
print(dataset.crs)
print(dataset.bounds)
print(dataset.width,dataset.height)
dataset_height=dataset.read(1)

def convert_projection(_coord_in,_epsg_in:str,_epsg_out:str):
    #_coord_in: (latitude,longitude)
    #_epsg_in: input projection (e.g. "EPSG:xxxx")
    #_epsg_out: output projection (e.g. "EPSG:xxxx")
    _t=Transformer.from_crs(_epsg_in,_epsg_out)
    _coord_out=_t.transform(_coord_in[0],_coord_in[1])
    return _coord_out

img_pos=convert_projection((img_lat,img_lon),"EPSG:4326",dataset.crs)#reprojected img_position
print(img_pos)
print(img_pos[0],img_pos[1])
print(dataset.index(img_pos[0],img_pos[1]))

#step 2: reproject all point into camera

img_eul_ang=[img_omega,img_phi,img_kappa]
Rot = R.from_euler("xyz", img_eul_ang, degrees=True).as_matrix()
print(type(Rot))
trans = [-img_pos[0],-img_pos[1],-img_alt]
K_intr = np.array([[img_fx,0,img_w/2],
               [0,img_fy,img_h/2],
               [0,0,1]])
#find pixel in +-125m with 10m step
for _x in np.arange(img_pos[0]-125,img_pos[0]+125,10):
    for _y in np.arange(img_pos[1]-125,img_pos[1]+125,10):
        _row,_col=dataset.index(_x,_y)
        #print(_x,_y,dataset_height[_row,_col])
        _pos=np.array([_x,_y,dataset_height[_row,_col]])
        uv=K_intr@(Rot@(_pos+trans))
        uv[:2]=uv[:2]/uv[2]
        print(uv)



#step 3: calculate depth

