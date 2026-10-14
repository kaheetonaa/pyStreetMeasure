#K=3x3 matrix
#extrinsic=[R|t] 4x3 matrix
#cam_coord in ecef-->enu raster local
#raster=wxh

#import libs
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.spatial.transform import Rotation as R
import numpy as np
import math

#step 1: check camera coord, raster coord, plot them
img_name="DJI_20260730113144_0092.jpg" #summerschool data
img_lat=45.96379205555556
img_lon=7.923466638888889
img_omega=-179.92986808165162
img_phi=-0.015833472534527618# +180 because the camera face down
img_kappa=167.2765519781928+90# +90 because of rotation
img_w=8192
img_h=5460
img_fx=5460.92311
img_fy=5460.92311
img_alt=2072.714
img_rel_alt=82.948

dataset = pd.DataFrame(pd.read_csv("LC3D/Punti_SS26_Totale.csv",delimiter=";"))#GPS_Data
print(dataset.columns)
print(dataset)

def convert_projection(_coord_in,_epsg_in:str,_epsg_out:str):
    #_coord_in: (latitude,longitude)
    #_epsg_in: input projection (e.g. "EPSG:xxxx")
    #_epsg_out: output projection (e.g. "EPSG:xxxx")
    _t=Transformer.from_crs(_epsg_in,_epsg_out)
    _coord_out=_t.transform(_coord_in[0],_coord_in[1])
    return _coord_out

img_pos=convert_projection((img_lat,img_lon),"EPSG:4326","EPSG:32632")#reprojected img_position
print(img_pos)
print(img_pos[0],img_pos[1])

#step 2: reproject all point into camera

img_eul_ang=[img_omega,img_phi,img_kappa]
Rot = R.from_euler("xyz", img_eul_ang, degrees=True).as_matrix()
trans = [-img_pos[0],-img_pos[1],-img_alt]
K_intr = np.array([[img_fx,0,img_w/2],
               [0,img_fy,img_h/2],
               [0,0,1]])

#step 3: calculate depth
depth_reference=np.array([[-9999,-9999,-9999]])
for _i in range(len(dataset.index)):
            _pos=np.array([dataset["Est [m]"][_i],dataset["Nord [m]"][_i],dataset["Quota Elliss. WGS84 [m]"][_i]])#Est [m];Nord [m];Quota Elliss. WGS84 [m]'
            uv=K_intr@(Rot@(_pos+trans))
            uv[:2]=uv[:2]/uv[2]
            uv=uv.reshape(1,3)
            depth_reference=np.append(depth_reference,uv,axis=0)

depth_reference=depth_reference[1:]
depth_reference=depth_reference[depth_reference[:,0]>0]
depth_reference=depth_reference[depth_reference[:,0]<img_w]
depth_reference=depth_reference[depth_reference[:,1]>0]
depth_reference=depth_reference[depth_reference[:,1]<img_h]
#depth_reference[:,1]=img_h-depth_reference[:,1]
print(depth_reference)

plt.scatter(depth_reference[:,0],img_h-depth_reference[:,1])
plt.show()

#step 4: export to npy
np.save("LC3D/height-ref.npy",depth_reference)
