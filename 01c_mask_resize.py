import cv2
import numpy as np
import subprocess
import pandas as pd
path="Piacenza/img/"
df=pd.DataFrame(pd.read_csv("Piacenza/piacenza-mapillary-groundtruth-link.csv"))
list_id=list(df["id"])
subprocess.run(["mkdir",path+"mask_road"])
for i in list_id:
    _id=str(i)
    img=cv2.imread(path+"mask-fixed/"+_id+"_colored_segmented.png")
    road_value=np.array([128,64,128])
    # ---- resize ------
    img=cv2.resize(img,(1036,518),interpolation=cv2.INTER_NEAREST)
    mask=cv2.inRange(img,road_value,road_value)
    np.save(path+"mask_road/"+_id+"_mask_resize.npy",mask)
