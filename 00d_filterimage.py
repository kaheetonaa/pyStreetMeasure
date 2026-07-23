import pandas as pd
import subprocess
import json,urllib.request,requests

df=pd.DataFrame(pd.read_csv("Piacenza/piacenza-mapillary-groundtruth-link.csv"))
path="Piacenza/img/original/"
list_id=list(df["id"])
subprocess.run(["mkdir",path+"selected"])
for i in range(len(list_id)):
    _id=str(list_id[i])
    try:
        subprocess.run(["cp",path+_id+".jpg",path+"selected/"+_id+".jpg"])
    except:
        print("failed copy "+ _id)
print("done filtering image!")

