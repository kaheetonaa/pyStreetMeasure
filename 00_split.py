import os
from PIL import Image
from exiftool import ExifToolHelper
import numpy as np
import pandas as pd
import shutil

#----- Data structure
#/path/
#|
#|______split-MDE.csv: metadata files of all images
#|
#|_______metadata.csv: metadata files of all images
#   |
#   |_____________MDE: images for MDE inference
#   |
#   |_______METASHAPE: images for photogrammetry


#image list
path='LC3D/img/images/'
images=[i for i in os.listdir(path) if i.endswith("jpg")]
try:
    os.mkdir(path+"MDE")
except:
    print("folder MDE exists")
try:
    os.mkdir(path+"METASHAPE")
except:
    print("folder METASHAPE exists")
print("total number of images",len(images))
if len(images)==0:
    quit()
#image for MDE

img_mde=pd.DataFrame(pd.read_csv(path+"split-MDE.csv"))
metadata=["XMP:GPSLatitude","XMP:RtkStdLat","XMP:GPSLongitude","XMP:RtkStdLon","XMP:AbsoluteAltitude","XMP:RtkStdHgt","XMP:GimbalYawDegree","XMP:GimbalRollDegree","XMP:GimbalPitchDegree","Composite:FocalLength35efl"]
metadata_export=[]

for i in images:
    #read image exif
    img=Image.open(path+i)
    with ExifToolHelper() as et:
        for d in et.get_metadata(path+i):
            lat,std_lat,lng,std_lng,_h,std_h,yaw,roll,pitch,f=[float(d[i]) for i in metadata]
    w=float(img.width)
    h=float(img.height)
    s_w=35.9
    s_h=24
    fx=f*w/35.9
    fy=f*h/24
    metadata_export+=[{"lat":lat,
                       "std_lat":std_lat,
                       "lng":lng,
                       "std_lng":std_lng,
                       "h":_h,
                       "std_h":std_h,
                       "yaw":yaw,
                       "roll":roll,
                       "pitch":pitch,
                       "f":f,
                       "w":w,
                       "h":h,
                       "s_w":s_w,
                       "s_h":s_h,
                       "fx":fx,
                       "fy":fy}]
    if i in list(img_mde["photo"]):
        #copy to MDE folder
        shutil.move(path+i,path+"MDE/"+i)

    else:
        #copy to photogrammetry folder
        shutil.move(path+i,path+"METASHAPE/"+i)
metadata_export=pd.DataFrame(metadata_export)
metadata_export.to_csv(path+"metadata.csv")
