import pandas as pd
import os
import subprocess

df=pd.DataFrame(pd.read_csv("zen_svi_out_mapillary/pids_urls.csv"))
name=df[df["camera_type"]=="spherical"]["id"]
perspective=df[df["camera_type"]=="perspective"]["id"]
for _n in name:
    try:
        os.remove("zen_svi_out_mapillary/mly_svi/batch_1/"+str(_n)+".png")
    finally:
        break
for _n in perspective:
    subprocess.run(["convert","zen_svi_out_mapillary/mly_svi/batch_1/"+str(_n)+".png","zen_svi_out_mapillary/mly_svi/batch_1/"+str(_n)+".jpg"])

subprocess.run("/home/kuquanghuy/vsfm/bin/VisualSFM")
