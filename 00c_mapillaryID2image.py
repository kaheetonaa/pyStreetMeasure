import pandas as pd
import subprocess
import json,urllib.request,requests
from exif import Image
<<<<<<< HEAD
<<<<<<< HEAD
from src import Equirec2Perspec as E2P
import cv2
import numpy as np

equirect_to_perspective=False
df=pd.DataFrame(pd.read_csv("LC3D/mapillary_02.csv"))
path="LC3D/img_02/"
=======

df=pd.DataFrame(pd.read_csv("LC3D/mapillary_01.csv"))
path="LC3D/img/"
>>>>>>> 78e47197e479dd1a2791c7c95229782b143f5fbf
=======

df=pd.DataFrame(pd.read_csv("LC3D/mapillary_01.csv"))
path="LC3D/img/"
>>>>>>> 78e47197e479dd1a2791c7c95229782b143f5fbf
subprocess.run(["mkdir",path])
pre_url="https://graph.mapillary.com/"
suff_url="?access_token=MLY|4463150933761310|5995ca3757fc4f9a9c8f5e96b2efaa03&fields=thumb_original_url"
list_id=list(df["id_mapillary"])
list_angle=list(df["angle"])
list_lat=list(df["y"])
list_lng=list(df["x"])

def decdeg2dms(dd):
    mult = -1 if dd < 0 else 1
    mnt,sec = divmod(abs(dd)*3600, 60)
    deg,mnt = divmod(mnt, 60)
    return (mult*deg, mult*mnt, mult*sec)


for i in range(len(list_id)):
    _id=list_id[i]
    url=pre_url+str(_id)+suff_url
    with urllib.request.urlopen(url) as _u:
        img_url=json.loads(_u.read().decode())["thumb_original_url"]
        print(img_url)
<<<<<<< HEAD
<<<<<<< HEAD
    with open(path+'{}.jpg'.format(str(i)), 'wb') as handler:
        image_data = requests.get(img_url, stream=True).raw
        if equirect_to_perspective==True:
            equ=np.asarray(bytearray(image_data.read()), dtype="uint8")
            equ=cv2.imdecode(equ, cv2.IMREAD_COLOR)
            equ=E2P.Equirectangular(equ)
            image_data= equ.GetPerspective(120, 180, 0, 1000, 1000)
            success, encoded_image = cv2.imencode('.jpg', image_data)
            image_data=encoded_image.tobytes()
=======
    with open(path+'{}.jpg'.format(str(_id)), 'wb') as handler:
        image_data = requests.get(img_url, stream=True).content
>>>>>>> 78e47197e479dd1a2791c7c95229782b143f5fbf
=======
    with open(path+'{}.jpg'.format(str(_id)), 'wb') as handler:
        image_data = requests.get(img_url, stream=True).content
>>>>>>> 78e47197e479dd1a2791c7c95229782b143f5fbf
        img_exif=Image(image_data)
        img_exif.gps_latitude=decdeg2dms(list_lat[i])
        img_exif.gps_longitude=decdeg2dms(list_lng[i])
        img_exif.gps_altitude=float(0)
        img_exif.gps_img_direction=list_angle[i]
        handler.write(img_exif.get_file())
    print(str(i+1) + " image(s) download in total " + str(len(list_id)))



