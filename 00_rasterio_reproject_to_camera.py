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
import argparse

#---parse parameters--------------------------------
parser=argparse.ArgumentParser(
                    prog='reproject DEM data to camera',
                    description='this piece of code would reproject ellipsoidal DEM data to known camera position. The inputs are image and raster DEM with camera position and rotation. The output is a single npy file. Developed for UAV cases',
                    add_help=True)
parser.add_argument("--dem",help="ellipsodal dem file path",type=str,default=0,required=True)
parser.add_argument("--lat",help="Latitude of camera pose",type=float,default=0,required=True)
parser.add_argument("--lon",help="Longitude of camera pose",type=float,default=0,required=True)
parser.add_argument("--omega",help="Yaw angle of camera pose in degree",type=float,default=0,required=True)
parser.add_argument("--phi",help="Pitch angle of camera pose in degree",type=float,default=0,required=True)
parser.add_argument("--kappa",help="Roll angle of camera pose in degree",type=float,default=0,required=True)
parser.add_argument("--w",help="Image width in pixel",type=float,default=0,required=True)
parser.add_argument("--h",help="Image height in pixel",type=float,default=0,required=True)
parser.add_argument("--fx",help="Focal length x",type=float,default=0,required=True)
parser.add_argument("--fy",help="Focal length y",type=float,default=0,required=True)
parser.add_argument("--height_abs",help="Absolute height of camera pose",type=float,default=-9999)
parser.add_argument("--height_rel",help="Relative height of camera pose",type=float,default=-9999)
parser.add_argument("--z_scale",help="inverse z scale by put -1",type=int,default=1)
parser.add_argument("--y_scale",help="inverse y scale by put -1",type=int,default=1)
parser.add_argument("--sample_size",help="sample_size",type=int,default=10)
parser.add_argument("--output",help="output npy path",type=str,required=True)
#TODO add algorithm to calculate from relative height of camera pose
args = parser.parse_args()

#----------------------------------------------------
#step 1: check camera coord, raster coord, plot them

img_lat=args.lat
img_lon=args.lon
img_omega=args.omega 
img_phi=args.phi
img_kappa=args.kappa
img_w=args.w
img_h=args.h
img_fx=args.fx
img_fy=args.fy
img_alt=args.height_abs
img_rel_alt=args.height_rel
dataset = rasterio.open(args.dem)
dataset_height=dataset.read(1)

def convert_projection(_coord_in,_epsg_in:str,_epsg_out:str):
    #_coord_in: (latitude,longitude)
    #_epsg_in: input projection (e.g. "EPSG:xxxx")
    #_epsg_out: output projection (e.g. "EPSG:xxxx")
    _t=Transformer.from_crs(_epsg_in,_epsg_out)
    _coord_out=_t.transform(_coord_in[0],_coord_in[1])
    return _coord_out

img_pos=convert_projection((img_lat,img_lon),"EPSG:4326",dataset.crs)#reprojected img_position

#----------------------------------------------------
#step 2: reproject all point into camera

img_eul_ang=[img_omega,img_phi,img_kappa]
Rot = R.from_euler("xyz", img_eul_ang, degrees=True).as_matrix()
trans = [img_pos[0],img_pos[1],img_alt]
inv_z=np.array([[1,0,0],[0,args.y_scale,0],[0,0,args.z_scale]])
K_intr = np.array([[img_fx,0,img_w/2],
               [0,img_fy,img_h/2],
               [0,0,1]])

#----------------------------------------------------
#step 3: calculate depth
#filter pixel in +-125m with 10m step
depth_reference=np.array([[-9999,-9999,-9999]])
for _x in np.arange(img_pos[0]-100,img_pos[0]+100,args.sample_size):
    for _y in np.arange(img_pos[1]-100,img_pos[1]+100,args.sample_size):
        _row,_col=dataset.index(_x,_y)
        if (dataset_height[_row,_col]>0):
            if (dataset_height[_row,_col]<3000):
                _pos=np.array([_x,_y,dataset_height[_row,_col]])
                uv=K_intr@(inv_z@(Rot@(_pos-trans)))
                uv[:2]=uv[:2]/uv[2]
                uv=uv.reshape(1,3)
                depth_reference=np.append(depth_reference,uv,axis=0)

depth_reference=depth_reference[1:]
depth_reference=depth_reference[depth_reference[:,0]>0]
depth_reference=depth_reference[depth_reference[:,0]<img_w]
depth_reference=depth_reference[depth_reference[:,1]>0]
depth_reference=depth_reference[depth_reference[:,1]<img_h]

print(depth_reference.min(),depth_reference.max())


fig, ax = plt.subplots()
scatter=ax.scatter(depth_reference[:,0],depth_reference[:,1],c=depth_reference[:,2],cmap="Greys")
ax.invert_yaxis()


cbar = fig.colorbar(scatter)

plt.show()

#----------------------------------------------------
#step 4: export to npy
np.save(args.output,depth_reference)
